"""SHAP-based feature importance with multi-explainer support.

This module provides SHAP value computation with automatic explainer selection
for tree-based, linear, and model-agnostic approaches.
"""

import warnings
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _detect_gpu_available() -> bool:
    """Detect if GPU acceleration is available for SHAP computations.

    GPU acceleration is currently supported only for TreeExplainer with:
    - NVIDIA GPU
    - CUDA 11.0+
    - cupy library installed

    Returns
    -------
    bool
        True if GPU is available and cupy is installed, False otherwise

    Notes
    -----
    This function checks for cupy availability as a proxy for GPU support.
    Even if a GPU is present, cupy must be installed for SHAP to use it.

    GPU acceleration provides 10-100x speedup for large datasets (>5K samples)
    but has overhead that makes it slower for small datasets (<5K samples).
    """
    try:
        import cupy as cp

        # Check if GPU is actually accessible
        _ = cp.cuda.Device(0)
        return True
    except (ImportError, RuntimeError):
        # ImportError: cupy not installed
        # RuntimeError: CUDA not available or no GPU found
        return False


def _get_explainer(
    model: Any,
    X_array: "NDArray[Any]",
    explainer_type: str = "auto",
    use_gpu: bool | str = "auto",
    background_data: Union["NDArray[Any]", None] = None,
    **explainer_kwargs: Any,
) -> tuple[Any, str, float]:
    """Select and create appropriate SHAP explainer for the given model.

    Implements automatic explainer selection with try-except cascade:
    1. TreeExplainer (fast, exact, tree models only)
    2. LinearExplainer (fast, exact, linear models only)
    3. KernelExplainer (slow, approximate, model-agnostic fallback)

    DeepExplainer is NOT included in auto-selection because it requires
    explicit background data specification. Use explainer_type="deep" explicitly.

    Parameters
    ----------
    model : Any
        Fitted model to explain
    X_array : np.ndarray
        Feature matrix for SHAP computation
    explainer_type : str, default "auto"
        Explainer type to use:
        - "auto": Try tree -> linear -> kernel (recommended)
        - "tree": TreeExplainer (tree models only)
        - "linear": LinearExplainer (linear models only)
        - "deep": DeepExplainer (neural networks, requires background_data)
        - "kernel": KernelExplainer (model-agnostic, slow)
    use_gpu : bool | str, default "auto"
        GPU acceleration mode (TreeExplainer only):
        - "auto": Use GPU if available and dataset large enough (>5K samples)
        - True: Force GPU usage (raises error if unavailable)
        - False: Force CPU usage
    background_data : np.ndarray | None, default None
        Background dataset for explainers that need it (Kernel, Deep).
        If None, will be auto-sampled from X_array for Kernel.
        Required for Deep explainer.
    **explainer_kwargs : Any
        Additional keyword arguments passed to explainer constructor

    Returns
    -------
    tuple[Any, str, float]
        - explainer: Initialized SHAP explainer instance
        - type_name: Name of explainer type used ("tree", "linear", "kernel", "deep")
        - ms_per_sample: Estimated milliseconds per sample for performance warnings

    Raises
    ------
    ImportError
        If shap library not installed
    ValueError
        If explainer_type is invalid or if auto-selection fails for all explainers
    RuntimeError
        If GPU requested but unavailable
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "SHAP library is not installed. Install with: pip install ml4t-diagnostic[ml] or: pip install shap>=0.41.0"
        ) from e

    # Validate explainer_type
    valid_types = {"auto", "tree", "linear", "deep", "kernel"}
    if explainer_type not in valid_types:
        raise ValueError(
            f"Invalid explainer_type '{explainer_type}'. Must be one of: {', '.join(sorted(valid_types))}"
        )

    # Handle GPU detection and configuration
    gpu_available = _detect_gpu_available()
    use_gpu_final = False

    if use_gpu == "auto":
        # Auto-detect: Use GPU if available AND dataset large enough
        n_samples = X_array.shape[0]
        use_gpu_final = gpu_available and n_samples >= 5000
    elif use_gpu is True:
        if not gpu_available:
            raise RuntimeError(
                "GPU requested (use_gpu=True) but GPU not available. "
                "Ensure NVIDIA GPU, CUDA 11.0+, and cupy are installed. "
                "Install with: pip install ml4t-diagnostic[gpu]"
            )
        use_gpu_final = True
    else:  # use_gpu is False
        use_gpu_final = False

    # Explicit explainer type requested
    if explainer_type != "auto":
        return _create_explainer_by_type(
            explainer_type=explainer_type,
            model=model,
            X_array=X_array,
            use_gpu=use_gpu_final,
            background_data=background_data,
            shap=shap,
            **explainer_kwargs,
        )

    # Auto-selection cascade: Tree -> Linear -> Kernel
    errors = []

    # Try TreeExplainer first (fastest, most common)
    try:
        tree_kwargs = {"feature_perturbation": "tree_path_dependent"}
        tree_kwargs.update(explainer_kwargs)  # User kwargs override defaults

        explainer = shap.TreeExplainer(model, **tree_kwargs)
        # GPU mode only for tree explainer
        if use_gpu_final and hasattr(explainer, "gpu"):
            setattr(explainer, "gpu", True)  # noqa: B010
        ms_per_sample = 5.0  # ~1-10ms typical
        return (explainer, "tree", ms_per_sample)
    except Exception as e:
        errors.append(f"TreeExplainer: {e}")

    # Try LinearExplainer second (fast, exact for linear models)
    try:
        explainer = shap.LinearExplainer(model, X_array, **explainer_kwargs)
        ms_per_sample = 75.0  # ~50-100ms typical
        return (explainer, "linear", ms_per_sample)
    except Exception as e:
        errors.append(f"LinearExplainer: {e}")

    # Try KernelExplainer as fallback (slow but model-agnostic)
    try:
        # Sample background data if not provided
        if background_data is None:
            background_data = _sample_background(X_array, max_samples=100, method="random")

        # Create prediction function wrapper to avoid LightGBM property issues
        if hasattr(model, "predict_proba"):
            # For binary classification, return probability of positive class
            def predict_fn(X):
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    return proba[:, 1]  # Binary: positive class
                return proba  # Multiclass: all classes
        else:
            predict_fn = model.predict

        explainer = shap.KernelExplainer(predict_fn, background_data, **explainer_kwargs)
        ms_per_sample = 5000.0  # ~1-10 seconds typical
        return (explainer, "kernel", ms_per_sample)
    except Exception as e:
        errors.append(f"KernelExplainer: {e}")

    # All explainers failed
    error_summary = "\n  - ".join(errors)
    raise ValueError(
        f"Failed to create explainer for model type {type(model).__name__}. "
        f"Tried tree, linear, and kernel explainers. Errors:\n  - {error_summary}\n"
        f"Consider using explainer_type='kernel' explicitly with custom background_data."
    )


def _create_explainer_by_type(
    explainer_type: str,
    model: Any,
    X_array: "NDArray[Any]",
    use_gpu: bool,
    background_data: Union["NDArray[Any]", None],
    shap: Any,
    **explainer_kwargs: Any,
) -> tuple[Any, str, float]:
    """Create specific explainer type (helper for _get_explainer).

    Parameters
    ----------
    explainer_type : str
        One of: "tree", "linear", "deep", "kernel"
    model : Any
        Fitted model
    X_array : np.ndarray
        Feature matrix
    use_gpu : bool
        Whether to use GPU (tree only)
    background_data : np.ndarray | None
        Background data for kernel/deep explainers
    shap : module
        Imported shap module
    **explainer_kwargs : Any
        Additional explainer arguments

    Returns
    -------
    tuple[Any, str, float]
        (explainer, type_name, ms_per_sample)

    Raises
    ------
    ValueError
        If explainer creation fails
    ImportError
        If deep learning dependencies not available
    """
    try:
        if explainer_type == "tree":
            # Set default feature_perturbation unless user overrides
            tree_kwargs = {"feature_perturbation": "tree_path_dependent"}
            tree_kwargs.update(explainer_kwargs)  # User kwargs override defaults

            explainer = shap.TreeExplainer(model, **tree_kwargs)
            if use_gpu and hasattr(explainer, "gpu"):
                explainer.gpu = True
            ms_per_sample = 5.0
            return (explainer, "tree", ms_per_sample)

        elif explainer_type == "linear":
            explainer = shap.LinearExplainer(model, X_array, **explainer_kwargs)
            ms_per_sample = 75.0
            return (explainer, "linear", ms_per_sample)

        elif explainer_type == "deep":
            if background_data is None:
                raise ValueError(
                    "DeepExplainer requires background_data parameter. "
                    "Provide a representative sample of your training data "
                    "(typically 100-1000 samples)."
                )
            try:
                explainer = shap.DeepExplainer(model, background_data, **explainer_kwargs)
            except ImportError as e:
                raise ImportError(
                    "DeepExplainer requires deep learning libraries (TensorFlow or PyTorch). "
                    "Install with: pip install ml4t-diagnostic[deep]"
                ) from e
            ms_per_sample = 500.0  # ~100ms-1s typical
            return (explainer, "deep", ms_per_sample)

        elif explainer_type == "kernel":
            if background_data is None:
                background_data = _sample_background(X_array, max_samples=100, method="random")

            # Create prediction function wrapper to avoid LightGBM property issues
            # For classifiers, use predict_proba if available (more informative)
            if hasattr(model, "predict_proba"):
                # For binary classification, return probability of positive class
                def predict_fn(X):
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:
                        return proba[:, 1]  # Binary: positive class
                    return proba  # Multiclass: all classes
            else:
                predict_fn = model.predict

            explainer = shap.KernelExplainer(predict_fn, background_data, **explainer_kwargs)
            ms_per_sample = 5000.0
            return (explainer, "kernel", ms_per_sample)

        else:
            raise ValueError(f"Unknown explainer_type: {explainer_type}")

    except Exception as e:
        raise ValueError(
            f"Failed to create {explainer_type.capitalize()}Explainer for model type {type(model).__name__}. Error: {e}"
        ) from e


def _sample_background(
    X_array: "NDArray[Any]", max_samples: int = 100, method: str = "random"
) -> "NDArray[Any]":
    """Sample background dataset for KernelExplainer.

    Background data represents "typical" feature values used as reference
    for computing SHAP values. Smaller backgrounds = faster computation.

    Parameters
    ----------
    X_array : np.ndarray
        Full feature matrix
    max_samples : int, default 100
        Maximum number of background samples
    method : str, default "random"
        Sampling method: "random" or "kmeans"

    Returns
    -------
    np.ndarray
        Background dataset (max_samples, n_features)

    Notes
    -----
    - Random: Fast, simple, works well for most cases
    - K-means: Better representation of data distribution, slower
    """
    n_samples = X_array.shape[0]

    if n_samples <= max_samples:
        return X_array

    if method == "random":
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        return X_array[idx]
    elif method == "kmeans":
        # K-means clustering for representative samples
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=max_samples, random_state=42, n_init=10)
            kmeans.fit(X_array)
            return kmeans.cluster_centers_
        except ImportError:
            # Fallback to random if sklearn not available
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, size=max_samples, replace=False)
            return X_array[idx]
    else:
        raise ValueError(f"Unknown sampling method: {method}. Use 'random' or 'kmeans'.")


def _estimate_computation_time(
    explainer_type: str,
    n_samples: int,
    ms_per_sample: float,
    performance_warning: bool = True,
) -> None:
    """Estimate SHAP computation time and issue warnings for slow explainers.

    Warns users before computationally expensive SHAP calculations to prevent
    unexpected long wait times, especially for KernelExplainer.

    Parameters
    ----------
    explainer_type : str
        Type of explainer being used ("tree", "linear", "kernel", "deep")
    n_samples : int
        Number of samples for SHAP computation
    ms_per_sample : float
        Estimated milliseconds per sample for this explainer type
    performance_warning : bool, default True
        Whether to issue performance warnings. Set to False to disable.
    """
    if not performance_warning:
        return

    # Only warn for KernelExplainer (1-10 seconds per sample)
    if explainer_type != "kernel":
        return

    # Compute estimates
    total_seconds = (n_samples * ms_per_sample) / 1000.0
    threshold_seconds = 10.0  # Warn if >10 seconds

    if total_seconds < threshold_seconds:
        return

    # Issue warning with time estimates
    time_str = _format_time(total_seconds)

    # Suggest max_samples=200 as reasonable default
    recommended_samples = 200
    if n_samples > recommended_samples:
        recommended_seconds = (recommended_samples * ms_per_sample) / 1000.0
        recommended_time_str = _format_time(recommended_seconds)

        warnings.warn(
            f"KernelExplainer is slow (~{int(ms_per_sample)}ms per sample).\n"
            f"Estimated time: ~{time_str} for {n_samples} samples.\n"
            f"Consider using max_samples={recommended_samples} "
            f"(estimated time: ~{recommended_time_str}).\n"
            f"Or use explainer_type='tree' or 'linear' for faster computation if model supports it.",
            UserWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"KernelExplainer is slow (~{int(ms_per_sample)}ms per sample).\n"
            f"Estimated time: ~{time_str} for {n_samples} samples.\n"
            f"Consider using explainer_type='tree' or 'linear' for faster computation if model supports it.",
            UserWarning,
            stacklevel=3,
        )


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Human-readable time string (e.g., "2 minutes", "1 hour 15 minutes")

    Examples
    --------
    >>> _format_time(45)
    '45 seconds'
    >>> _format_time(120)
    '2 minutes'
    >>> _format_time(3665)
    '1 hour 1 minute'
    """
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        hours = int(seconds / 3600)
        remaining_minutes = int((seconds % 3600) / 60)
        if remaining_minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            return f"{hours} hour{'s' if hours != 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"


def compute_shap_importance(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    feature_names: list[str] | None = None,
    check_additivity: bool = True,
    max_samples: int | None = None,
    explainer_type: str = "auto",
    use_gpu: bool | str = "auto",
    background_data: Union["NDArray[Any]", None] = None,
    explainer_kwargs: dict | None = None,
    show_progress: bool = False,
    performance_warning: bool = True,
) -> dict[str, Any]:
    """Compute SHAP (SHapley Additive exPlanations) values and aggregate to feature importance.

    SHAP values provide a unified measure of feature importance based on Shapley values
    from cooperative game theory. Each feature's contribution to a prediction is
    calculated by considering all possible feature coalitions, satisfying key
    properties like additivity and consistency.

    **Key advantages over MDI and PFI**:

    - **Theoretically sound**: Based on game theory (Shapley values)
    - **Consistent**: Removing a feature always decreases its importance
    - **Local explanations**: Provides per-prediction feature contributions
    - **Interaction-aware**: Accounts for feature interactions naturally
    - **Unbiased**: No bias toward high-cardinality features (unlike MDI)
    - **Model-agnostic**: Works with ANY sklearn-compatible model (v1.1+)

    **Multi-Explainer Support**:

    This function automatically selects the best SHAP explainer for your model:

    - **TreeExplainer**: Fast, exact computation for tree-based models
    - **LinearExplainer**: Fast, exact computation for linear models
    - **KernelExplainer**: Model-agnostic fallback (slower but universal)
    - **DeepExplainer**: Optimized for neural networks (TensorFlow/PyTorch)

    Parameters
    ----------
    model : Any
        Fitted model compatible with SHAP explainers.
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix for SHAP computation (typically test/validation set)
        Shape: (n_samples, n_features)
    feature_names : list[str] | None, default None
        Feature names for labeling. If None, uses column names from DataFrame
        or generates numeric names for arrays
    check_additivity : bool, default True
        Verify that SHAP values sum to model predictions (sanity check).
        Only supported by TreeExplainer. Disable for speed if you trust the
        implementation.
    max_samples : int | None, default None
        Maximum number of samples to compute SHAP values for.
    explainer_type : str, default 'auto'
        SHAP explainer to use:
        - 'auto': Automatic selection (Tree -> Linear -> Kernel cascade)
        - 'tree': Force TreeExplainer
        - 'linear': Force LinearExplainer
        - 'kernel': Force KernelExplainer
        - 'deep': Force DeepExplainer
    use_gpu : Union[bool, str], default 'auto'
        Enable GPU acceleration for SHAP computation
    background_data : np.ndarray | None, default None
        Background dataset for KernelExplainer
    explainer_kwargs : dict | None, default None
        Additional keyword arguments passed to the explainer constructor
    show_progress : bool, default False
        Show progress bar for SHAP computation (requires tqdm)
    performance_warning : bool, default True
        Issue warning if computation will take >10 seconds

    Returns
    -------
    dict[str, Any]
        Dictionary with SHAP importance results:
        - shap_values: SHAP values array, shape (n_samples, n_features)
        - importances: Mean absolute SHAP values per feature (sorted descending)
        - feature_names: Feature labels (sorted by importance)
        - base_value: Expected model output (average prediction)
        - n_features: Number of features
        - n_samples: Number of samples used for SHAP computation
        - model_type: Type of model used
        - explainer_type: Which explainer was used
        - additivity_verified: Whether additivity check passed

    Raises
    ------
    ImportError
        If shap library not installed
    ValueError
        If model is not supported by specified explainer
    RuntimeError
        If SHAP computation fails
    """
    # Check if shap is installed
    try:
        import shap  # noqa: F401 (availability check)
    except ImportError as e:
        raise ImportError(
            "SHAP library is not installed. Install with: pip install ml4t-diagnostic[ml] or: pip install shap>=0.43.0"
        ) from e

    # Convert X to appropriate format
    if isinstance(X, pl.DataFrame):
        X_array = X.to_numpy()
        if feature_names is None:
            feature_names = X.columns
    elif isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_array = np.asarray(X)

    # Validate shape before accessing shape[1]
    if X_array.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X_array.shape}")

    # Set default feature names if needed (after shape validation)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    # Ensure feature_names is a list
    if feature_names is not None:
        feature_names = list(feature_names)

    n_samples_full, n_features = X_array.shape

    # Subsample if requested
    if max_samples is not None and n_samples_full > max_samples:
        # Use random sampling for representative subset
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_samples_full, size=max_samples, replace=False)
        X_array = X_array[sample_idx]
        n_samples = max_samples
    else:
        n_samples = n_samples_full

    # Validate feature names length
    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not match number of features in X ({n_features})"
        )

    # Get appropriate explainer (auto-selects or uses explicit type)
    if explainer_kwargs is None:
        explainer_kwargs = {}

    explainer, explainer_type_used, ms_per_sample = _get_explainer(
        model=model,
        X_array=X_array,
        explainer_type=explainer_type,
        use_gpu=use_gpu,
        background_data=background_data,
        **explainer_kwargs,
    )

    # Issue performance warning if needed
    _estimate_computation_time(
        explainer_type=explainer_type_used,
        n_samples=n_samples,
        ms_per_sample=ms_per_sample,
        performance_warning=performance_warning,
    )

    # Compute SHAP values with optional progress bar
    try:
        # Only TreeExplainer supports check_additivity parameter
        shap_kwargs = {}
        if explainer_type_used == "tree":
            shap_kwargs["check_additivity"] = check_additivity

        if show_progress:
            try:
                from tqdm.auto import tqdm

                # Wrap computation with progress bar for slow explainers
                if explainer_type_used == "kernel":
                    # For kernel, show progress
                    with tqdm(total=n_samples, desc="Computing SHAP values") as pbar:
                        shap_values_raw = explainer.shap_values(X_array, **shap_kwargs)
                        pbar.update(n_samples)
                else:
                    # For tree/linear/deep, just compute (fast enough)
                    shap_values_raw = explainer.shap_values(X_array, **shap_kwargs)
            except ImportError:
                # tqdm not available, compute without progress bar
                shap_values_raw = explainer.shap_values(X_array, **shap_kwargs)
        else:
            shap_values_raw = explainer.shap_values(X_array, **shap_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute SHAP values with {explainer_type_used}Explainer. "
            f"Model type: {type(model).__name__}. Error: {e}"
        ) from e

    # Handle binary classification (returns list of arrays OR 3D array)
    if isinstance(shap_values_raw, list):
        if len(shap_values_raw) == 2:
            # Binary classification (older SHAP versions)
            shap_values = shap_values_raw[1]
        else:
            # Multiclass - use first class for importance
            shap_values = shap_values_raw[0]
    else:
        shap_values = shap_values_raw
        # Handle 3D array for binary/multiclass (newer SHAP versions)
        if shap_values.ndim == 3:
            if shap_values.shape[2] == 2:
                # Binary classification: take positive class (index 1)
                shap_values = shap_values[:, :, 1]
            else:
                # Multiclass: aggregate across classes (mean absolute)
                shap_values = np.mean(np.abs(shap_values), axis=2)

    # Validate SHAP values shape
    if shap_values.shape != (n_samples, n_features):
        raise RuntimeError(
            f"Unexpected SHAP values shape: {shap_values.shape}, expected ({n_samples}, {n_features})"
        )

    # Compute feature importance as mean absolute SHAP value
    importances = np.mean(np.abs(shap_values), axis=0)

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]

    # Get base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, list | np.ndarray):
        # For binary/multiclass, take positive class or first class
        base_value = base_value[1] if len(base_value) == 2 else base_value[0]

    # Determine model type
    model_type = f"{type(model).__module__}.{type(model).__name__}"

    return {
        "shap_values": shap_values,
        "importances": importances[sorted_idx],
        "feature_names": [feature_names[i] for i in sorted_idx],
        "base_value": float(base_value),
        "n_features": n_features,
        "n_samples": n_samples,
        "model_type": model_type,
        "explainer_type": explainer_type_used,
        "additivity_verified": check_additivity,
    }

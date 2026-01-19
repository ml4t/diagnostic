# Test Runtime Catalog

Generated: 2026-01-05 11:35:00

## Summary

| Suite | Tests | Runtime |
|-------|------:|--------:|
| Fast (`pytest -m "not slow"`) | 2,785 | ~97s |
| Slow (`pytest -m slow`) | 850 | ~20 min |
| **Total** | 3,635 | ~22 min |

## Slow Test Files (marked with `pytestmark = pytest.mark.slow`)

These 9 files contain tests that timeout or take >60s. They are excluded from
the default test run. Run with: `pytest -m slow`

| File | Reason |
|:-----|:-------|
| `test_benchmarks/test_performance.py` | Benchmark tests with pytest-benchmark |
| `test_evaluation/test_ras_validation.py` | Monte Carlo simulations for RAS |
| `test_evaluation/test_domain_classifier.py` | ML model training |
| `test_evaluation/test_multi_signal_week2.py` | Multi-signal analysis |
| `test_evaluation/test_interaction_summary.py` | SHAP interaction computation |
| `test_evaluation/test_multi_signal_e2e.py` | Large-scale E2E tests |
| `test_evaluation/test_multi_signal_performance.py` | Signal analysis benchmarks |
| `test_backends/test_streaming.py` | 150K sample streaming tests |
| `test_evaluation/test_h_statistic.py` | H-statistic with RF models |

## Running Tests

```bash
# Fast tests only (CI default) - ~97 seconds
pytest -m "not slow"

# Slow tests only (nightly/release)
pytest -m slow

# All tests
pytest
```

## Medium-Speed Files (30-60s)

These files are included in fast suite but are notable:

| Runtime | Tests | File |
|--------:|------:|:-----|
| 49.5s | 86 | tests/test_evaluation/test_shap_importance.py |
| 42.1s | 36 | tests/test_visualization/test_report_generation.py |
| 32.9s | 264 | tests/test_evaluation/test_metrics.py |
| 30.3s | 89 | tests/test_evaluation/test_signal_analysis.py |

## Performance Notes

- Original runtime (all tests): >27 min (with 7 timeouts at 2 min each)
- Fast suite: ~97 seconds (2,785 tests)
- Achieved 17x speedup for CI by excluding slow tests

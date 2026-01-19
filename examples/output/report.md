# Feature Diagnostics: test_signal

**Generated**: 2025-11-03 21:57:25
**Observations**: 500
**Health Score**: 1.00/1.00

## Test Summary

| Module          | Test           |   Statistic |    P-Value | Result              |
|:----------------|:---------------|------------:|-----------:|:--------------------|
| Stationarity    | ADF            |  -22.3438   |   0        | Stationary          |
| Stationarity    | KPSS           |    0.101963 |   0.1      | Stationary          |
| Stationarity    | PP             |  -22.3723   |   0        | Stationary          |
| Stationarity    | Consensus      |  nan        | nan        | strong_stationary   |
| Autocorrelation | ACF            |  nan        | nan        | 0 significant lags  |
| Autocorrelation | PACF           |  nan        | nan        | 1 significant lags  |
| Autocorrelation | Consensus      |  nan        | nan        | Has autocorrelation |
| Volatility      | ARCH-LM        |    5.58608  |   0.935495 | No ARCH effects     |
| Volatility      | Consensus      |  nan        | nan        | No clustering       |
| Distribution    | Skewness       |    0.180164 | nan        | Not significant     |
| Distribution    | Kurtosis       |    0.27106  | nan        | Normal kurtosis     |
| Distribution    | Jarque-Bera    |    4.0581   |   0.13146  | Normal              |
| Distribution    | Shapiro-Wilk   |    0.996701 |   0.401269 | Normal              |
| Distribution    | Hill Estimator |    4.12569  | nan        | Thin                |
| Distribution    | Recommended    |  nan        | nan        | normal              |

## Stationarity Analysis

**Consensus**: strong_stationary

- **ADF**: statistic=-22.3438, p-value=0.0000
- **KPSS**: statistic=0.1020, p-value=0.1000
- **PP**: statistic=-22.3723, p-value=0.0000

## Autocorrelation Analysis

- **Significant ACF lags**: 0
- **Significant PACF lags**: 1
- **Suggested ARIMA order**: (0, 0, 0)
- **White noise**: No

## Volatility Analysis

- **Volatility clustering**: No
- **ARCH-LM**: statistic=5.5861, p-value=0.9355

## Distribution Analysis

- **Recommended distribution**: normal
- **Is normal**: Yes
- **Mean**: 0.006838
- **Std Dev**: 0.981253
- **Skewness**: 0.1802 (not significant)
- **Kurtosis**: 0.2711 (not significant)
- **Jarque-Bera**: statistic=4.0581, p-value=0.1315

## Recommendations

1. Feature has significant autocorrelation up to lag 8. Consider AR/MA modeling or including lagged values as features.

| Ticker   | Model                           |   Target Alpha | Empirical Coverage   | Coverage Error   |   Sharpness |   Pinball Loss |
|:---------|:--------------------------------|---------------:|:---------------------|:-----------------|------------:|---------------:|
| ^NSEI    | 1. Historical Simulation (252d) |           0.05 | 94.32%               | 0.68%            |      0.0148 |       0.001224 |
| ^NSEI    | 2. RiskMetrics EWMA (0.94)      |           0.05 | 94.20%               | 0.80%            |      0.0151 |       0.001138 |
| ^NSEI    | 3. GARCH(1,1)-Normal            |           0.05 | 94.84%               | 0.16%            |      0.0154 |       0.001138 |
| ^NSEI    | 4. GARCH(1,1)-Student-t         |           0.05 | 94.59%               | 0.41%            |      0.0148 |       0.001139 |
| ^NSEI    | 5. Uncalibrated LightGBM        |           0.05 | 93.20%               | 1.80%            |      0.0137 |       0.001238 |
| ^NSEI    | 6. LightGBM + Sliding CQR       |           0.05 | 94.69%               | 0.31%            |      0.0162 |       0.001226 |
| ^NSEI    | 7. LightGBM + Online ACI        |           0.05 | 94.90%               | 0.10%            |      0.0178 |       0.001218 |
| ^NSEI    | 8. LightGBM + Vol-Scaled ACI    |           0.05 | 94.93%               | 0.07%            |      0.0188 |       0.001248 |
| ^NSEI    | 1. Historical Simulation (252d) |           0.01 | 98.24%               | 0.76%            |      0.0258 |       0.000444 |
| ^NSEI    | 2. RiskMetrics EWMA (0.94)      |           0.01 | 98.03%               | 0.97%            |      0.0214 |       0.000372 |
| ^NSEI    | 3. GARCH(1,1)-Normal            |           0.01 | 98.30%               | 0.70%            |      0.0217 |       0.000368 |
| ^NSEI    | 4. GARCH(1,1)-Student-t         |           0.01 | 98.63%               | 0.37%            |      0.0231 |       0.000364 |
| ^NSEI    | 5. Uncalibrated LightGBM        |           0.01 | 97.81%               | 1.19%            |      0.0222 |       0.00048  |
| ^NSEI    | 6. LightGBM + Sliding CQR       |           0.01 | 98.88%               | 0.12%            |      0.0315 |       0.000435 |
| ^NSEI    | 7. LightGBM + Online ACI        |           0.01 | 98.60%               | 0.40%            |      0.032  |       0.000436 |
| ^NSEI    | 8. LightGBM + Vol-Scaled ACI    |           0.01 | 98.72%               | 0.28%            |      0.0311 |       0.000413 |
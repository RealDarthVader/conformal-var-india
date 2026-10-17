# Market Regimes VaR Backtest Scorecard (^NSEI)

| Ticker   | Regime                |   Alpha | Model                        | Coverage   |   Violations |   Sharpness |   Pinball Loss |   Kupiec p-val |   Christoffersen p-val | Basel Zone   |
|----------|-----------------------|---------|------------------------------|------------|--------------|-------------|----------------|----------------|------------------------|--------------|
| ^NSEI    | full_sample_10y       |    0.05 | Historical Simulation (252d) | 94.32%     |          187 |      0.0148 |       0.001224 |         0.08   |                 0.001  | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | RiskMetrics EWMA (0.94)      | 94.20%     |          191 |      0.0151 |       0.001138 |         0.0397 |                 0.7723 | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | GARCH(1,1)-Normal            | 94.84%     |          170 |      0.0154 |       0.001138 |         0.6704 |                 0.6701 | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | GARCH(1,1)-Student-t         | 94.59%     |          178 |      0.0148 |       0.001139 |         0.2918 |                 0.646  | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | Uncalibrated LightGBM        | 93.20%     |          224 |      0.0137 |       0.001238 |         0      |                 0.0451 | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | LightGBM + Sliding CQR       | 94.69%     |          175 |      0.0162 |       0.001226 |         0.4125 |                 0.2243 | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | LightGBM + Online ACI        | 94.90%     |          168 |      0.0178 |       0.001218 |         0.7895 |                 0.0709 | Red          |
| ^NSEI    | full_sample_10y       |    0.05 | LightGBM + Vol-Scaled ACI    | 94.93%     |          167 |      0.0188 |       0.001248 |         0.8513 |                 0.0648 | Red          |
| ^NSEI    | full_sample_10y       |    0.01 | Historical Simulation (252d) | 98.24%     |           58 |      0.0258 |       0.000444 |         0.0001 |                 0.1038 | Green        |
| ^NSEI    | full_sample_10y       |    0.01 | RiskMetrics EWMA (0.94)      | 98.03%     |           65 |      0.0214 |       0.000372 |         0      |                 0.7909 | Yellow       |
| ^NSEI    | full_sample_10y       |    0.01 | GARCH(1,1)-Normal            | 98.30%     |           56 |      0.0217 |       0.000368 |         0.0002 |                 0.9609 | Green        |
| ^NSEI    | full_sample_10y       |    0.01 | GARCH(1,1)-Student-t         | 98.63%     |           45 |      0.0231 |       0.000364 |         0.0452 |                 0.6477 | Green        |
| ^NSEI    | full_sample_10y       |    0.01 | Uncalibrated LightGBM        | 97.81%     |           72 |      0.0222 |       0.00048  |         0      |                 0.0956 | Yellow       |
| ^NSEI    | full_sample_10y       |    0.01 | LightGBM + Sliding CQR       | 98.88%     |           37 |      0.0315 |       0.000435 |         0.4846 |                 0.0713 | Green        |
| ^NSEI    | full_sample_10y       |    0.01 | LightGBM + Online ACI        | 98.60%     |           46 |      0.032  |       0.000436 |         0.0308 |                 0.6756 | Green        |
| ^NSEI    | full_sample_10y       |    0.01 | LightGBM + Vol-Scaled ACI    | 98.72%     |           42 |      0.0311 |       0.000413 |         0.1276 |                 0.5657 | Green        |
| ^NSEI    | calm_bull             |    0.05 | Historical Simulation (252d) | 97.77%     |            6 |      0.0111 |       0.000671 |         0.0198 |                 0.1099 | Yellow       |
| ^NSEI    | calm_bull             |    0.05 | RiskMetrics EWMA (0.94)      | 95.17%     |           13 |      0.0097 |       0.000635 |         0.8993 |                 0.1379 | Red          |
| ^NSEI    | calm_bull             |    0.05 | GARCH(1,1)-Normal            | 98.88%     |            3 |      0.0122 |       0.000704 |         0.0004 |                 0.7944 | Green        |
| ^NSEI    | calm_bull             |    0.05 | GARCH(1,1)-Student-t         | 98.14%     |            5 |      0.0117 |       0.000687 |         0.007  |                 0.6628 | Yellow       |
| ^NSEI    | calm_bull             |    0.05 | Uncalibrated LightGBM        | 99.26%     |            2 |      0.0135 |       0.000737 |         0.0001 |                 0.8623 | Green        |
| ^NSEI    | calm_bull             |    0.05 | LightGBM + Sliding CQR       | 98.88%     |            3 |      0.0122 |       0.000696 |         0.0004 |                 0.7944 | Green        |
| ^NSEI    | calm_bull             |    0.05 | LightGBM + Online ACI        | 96.65%     |            9 |      0.0106 |       0.00066  |         0.1864 |                 0.2911 | Yellow       |
| ^NSEI    | calm_bull             |    0.05 | LightGBM + Vol-Scaled ACI    | 96.65%     |            9 |      0.0103 |       0.000645 |         0.1864 |                 0.2911 | Yellow       |
| ^NSEI    | calm_bull             |    0.01 | Historical Simulation (252d) | 99.63%     |            1 |      0.0183 |       0.000197 |         0.2348 |                 0.931  | Green        |
| ^NSEI    | calm_bull             |    0.01 | RiskMetrics EWMA (0.94)      | 99.26%     |            2 |      0.0138 |       0.000167 |         0.6578 |                 0.8623 | Green        |
| ^NSEI    | calm_bull             |    0.01 | GARCH(1,1)-Normal            | 99.63%     |            1 |      0.0173 |       0.000185 |         0.2348 |                 0.931  | Green        |
| ^NSEI    | calm_bull             |    0.01 | GARCH(1,1)-Student-t         | 100.00%    |            0 |      0.0187 |       0.000198 |         0.0201 |                 1      | Green        |
| ^NSEI    | calm_bull             |    0.01 | Uncalibrated LightGBM        | 100.00%    |            0 |      0.0219 |       0.00023  |         0.0201 |                 1      | Green        |
| ^NSEI    | calm_bull             |    0.01 | LightGBM + Sliding CQR       | 100.00%    |            0 |      0.0201 |       0.000212 |         0.0201 |                 1      | Green        |
| ^NSEI    | calm_bull             |    0.01 | LightGBM + Online ACI        | 98.88%     |            3 |      0.0159 |       0.000196 |         0.852  |                 0.7944 | Green        |
| ^NSEI    | calm_bull             |    0.01 | LightGBM + Vol-Scaled ACI    | 99.63%     |            1 |      0.0169 |       0.000187 |         0.2348 |                 0.931  | Green        |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | Historical Simulation (252d) | 90.58%     |           13 |      0.0122 |       0.001159 |         0.0328 |                 0.0206 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | RiskMetrics EWMA (0.94)      | 92.75%     |           10 |      0.0144 |       0.001049 |         0.2551 |                 0.0226 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | GARCH(1,1)-Normal            | 92.75%     |           10 |      0.014  |       0.001037 |         0.2551 |                 0.0226 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | GARCH(1,1)-Student-t         | 92.75%     |           10 |      0.0137 |       0.001045 |         0.2551 |                 0.0226 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | Uncalibrated LightGBM        | 92.03%     |           11 |      0.0129 |       0.001035 |         0.1389 |                 0.2537 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | LightGBM + Sliding CQR       | 89.86%     |           14 |      0.0123 |       0.001133 |         0.0143 |                 0.5413 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | LightGBM + Online ACI        | 94.20%     |            8 |      0.0188 |       0.00107  |         0.6749 |                 0.0464 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.05 | LightGBM + Vol-Scaled ACI    | 94.20%     |            8 |      0.0191 |       0.001157 |         0.6749 |                 0.0464 | Red          |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | Historical Simulation (252d) | 97.10%     |            4 |      0.021  |       0.000365 |         0.0683 |                 0.0848 | Yellow       |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | RiskMetrics EWMA (0.94)      | 97.83%     |            3 |      0.0204 |       0.000269 |         0.2304 |                 0.0392 | Yellow       |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | GARCH(1,1)-Normal            | 98.55%     |            2 |      0.0198 |       0.00027  |         0.6191 |                 0.0121 | Green        |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | GARCH(1,1)-Student-t         | 98.55%     |            2 |      0.0206 |       0.000264 |         0.6191 |                 0.0121 | Green        |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | Uncalibrated LightGBM        | 97.10%     |            4 |      0.0195 |       0.000287 |         0.0683 |                 0.0848 | Yellow       |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | LightGBM + Sliding CQR       | 97.10%     |            4 |      0.022  |       0.000382 |         0.0683 |                 0.0017 | Yellow       |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | LightGBM + Online ACI        | 97.10%     |            4 |      0.0232 |       0.000329 |         0.0683 |                 0.0848 | Yellow       |
| ^NSEI    | ilfs_liquidity_crisis |    0.01 | LightGBM + Vol-Scaled ACI    | 97.10%     |            4 |      0.0226 |       0.000314 |         0.0683 |                 0.0848 | Yellow       |
| ^NSEI    | covid_crash           |    0.05 | Historical Simulation (252d) | 82.83%     |           17 |      0.02   |       0.006099 |         0      |                 0.3954 | Red          |
| ^NSEI    | covid_crash           |    0.05 | RiskMetrics EWMA (0.94)      | 89.90%     |           10 |      0.0426 |       0.004607 |         0.0395 |                 0.9257 | Red          |
| ^NSEI    | covid_crash           |    0.05 | GARCH(1,1)-Normal            | 88.89%     |           11 |      0.0349 |       0.00439  |         0.0154 |                 0.8955 | Red          |
| ^NSEI    | covid_crash           |    0.05 | GARCH(1,1)-Student-t         | 87.88%     |           12 |      0.0325 |       0.004513 |         0.0055 |                 0.7255 | Red          |
| ^NSEI    | covid_crash           |    0.05 | Uncalibrated LightGBM        | 79.80%     |           20 |      0.0165 |       0.005992 |         0      |                 0.4871 | Red          |
| ^NSEI    | covid_crash           |    0.05 | LightGBM + Sliding CQR       | 85.86%     |           14 |      0.0252 |       0.005292 |         0.0006 |                 0.4349 | Red          |
| ^NSEI    | covid_crash           |    0.05 | LightGBM + Online ACI        | 91.92%     |            8 |      0.0604 |       0.005173 |         0.1948 |                 0.5724 | Red          |
| ^NSEI    | covid_crash           |    0.05 | LightGBM + Vol-Scaled ACI    | 92.93%     |            7 |      0.0891 |       0.005897 |         0.3722 |                 0.3378 | Red          |
| ^NSEI    | covid_crash           |    0.01 | Historical Simulation (252d) | 90.91%     |            9 |      0.0538 |       0.003187 |         0      |                 0.7456 | Red          |
| ^NSEI    | covid_crash           |    0.01 | RiskMetrics EWMA (0.94)      | 92.93%     |            7 |      0.0602 |       0.00221  |         0.0001 |                 0.3378 | Red          |
| ^NSEI    | covid_crash           |    0.01 | GARCH(1,1)-Normal            | 91.92%     |            8 |      0.0494 |       0.002059 |         0      |                 0.2657 | Red          |
| ^NSEI    | covid_crash           |    0.01 | GARCH(1,1)-Student-t         | 91.92%     |            8 |      0.0497 |       0.002064 |         0      |                 0.2657 | Red          |
| ^NSEI    | covid_crash           |    0.01 | Uncalibrated LightGBM        | 83.84%     |           16 |      0.0213 |       0.004664 |         0      |                 0.6826 | Red          |
| ^NSEI    | covid_crash           |    0.01 | LightGBM + Sliding CQR       | 92.93%     |            7 |      0.068  |       0.002317 |         0.0001 |                 0.3378 | Red          |
| ^NSEI    | covid_crash           |    0.01 | LightGBM + Online ACI        | 93.94%     |            6 |      0.1045 |       0.002116 |         0.0006 |                 0.4205 | Red          |
| ^NSEI    | covid_crash           |    0.01 | LightGBM + Vol-Scaled ACI    | 95.96%     |            4 |      0.1139 |       0.001554 |         0.022  |                 0.6143 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | Historical Simulation (252d) | 94.35%     |           14 |      0.0176 |       0.001326 |         0.6476 |                 0.1945 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | RiskMetrics EWMA (0.94)      | 93.95%     |           15 |      0.0176 |       0.00129  |         0.4627 |                 0.9219 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | GARCH(1,1)-Normal            | 93.55%     |           16 |      0.0182 |       0.00129  |         0.3145 |                 0.9693 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | GARCH(1,1)-Student-t         | 93.15%     |           17 |      0.0173 |       0.001286 |         0.2034 |                 0.4478 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | Uncalibrated LightGBM        | 93.95%     |           15 |      0.0204 |       0.001472 |         0.4627 |                 0.2844 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | LightGBM + Sliding CQR       | 95.56%     |           11 |      0.0234 |       0.001472 |         0.6777 |                 0.4984 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | LightGBM + Online ACI        | 94.76%     |           13 |      0.0226 |       0.001498 |         0.8623 |                 0.7042 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.05 | LightGBM + Vol-Scaled ACI    | 95.16%     |           12 |      0.0235 |       0.001503 |         0.9067 |                 0.5988 | Red          |
| ^NSEI    | rate_hike_tightening  |    0.01 | Historical Simulation (252d) | 98.79%     |            3 |      0.0284 |       0.000377 |         0.748  |                 0.7859 | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | RiskMetrics EWMA (0.94)      | 98.39%     |            4 |      0.0249 |       0.000391 |         0.373  |                 0.7167 | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | GARCH(1,1)-Normal            | 98.39%     |            4 |      0.0257 |       0.000387 |         0.373  |                 0.7167 | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | GARCH(1,1)-Student-t         | 98.79%     |            3 |      0.0279 |       0.000379 |         0.748  |                 0.7859 | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | Uncalibrated LightGBM        | 98.39%     |            4 |      0.0307 |       0.000428 |         0.373  |                 0.7167 | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | LightGBM + Sliding CQR       | 100.00%    |            0 |      0.0407 |       0.000409 |         0.0256 |                 1      | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | LightGBM + Online ACI        | 98.79%     |            3 |      0.0381 |       0.000417 |         0.748  |                 0.7859 | Green        |
| ^NSEI    | rate_hike_tightening  |    0.01 | LightGBM + Vol-Scaled ACI    | 99.19%     |            2 |      0.0352 |       0.00038  |         0.7512 |                 0.8566 | Green        |

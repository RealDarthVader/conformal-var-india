# YESBANK.NS Left-Tail Collapse Stress Test (2018–2020)

| Ticker     | Regime                     |   Alpha | Model                        | Coverage   |   Violations |   Sharpness |   Pinball Loss |   Kupiec p-val |   Christoffersen p-val | Basel Zone   |
|------------|----------------------------|---------|------------------------------|------------|--------------|-------------|----------------|----------------|------------------------|--------------|
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | Historical Simulation (252d) | 92.26%     |           57 |      0.0695 |       0.007485 |         0.0015 |                 0.0032 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | RiskMetrics EWMA (0.94)      | 95.38%     |           34 |      0.0837 |       0.007091 |         0.6316 |                 0.7308 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | GARCH(1,1)-Normal            | 95.65%     |           32 |      0.0752 |       0.006689 |         0.4068 |                 0.2111 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | GARCH(1,1)-Student-t         | 93.34%     |           49 |      0.0703 |       0.007021 |         0.049  |                 0.1403 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | Uncalibrated LightGBM        | 85.33%     |          108 |      0.039  |       0.008356 |         0      |                 0      | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | LightGBM + Sliding CQR       | 93.75%     |           46 |      0.0724 |       0.007002 |         0.1335 |                 0.2211 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | LightGBM + Online ACI        | 95.79%     |           31 |      0.088  |       0.00681  |         0.3139 |                 0.0435 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.05 | LightGBM + Vol-Scaled ACI    | 95.52%     |           33 |      0.0993 |       0.007501 |         0.5133 |                 0.0146 | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | Historical Simulation (252d) | 97.96%     |           15 |      0.124  |       0.003266 |         0.0131 |                 0.0333 | Yellow       |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | RiskMetrics EWMA (0.94)      | 97.55%     |           18 |      0.1183 |       0.003361 |         0.0009 |                 0.4557 | Yellow       |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | GARCH(1,1)-Normal            | 97.96%     |           15 |      0.1063 |       0.003364 |         0.0131 |                 0.3062 | Yellow       |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | GARCH(1,1)-Student-t         | 98.37%     |           12 |      0.1308 |       0.003292 |         0.1152 |                 0.184  | Green        |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | Uncalibrated LightGBM        | 92.39%     |           56 |      0.0732 |       0.005047 |         0      |                 0      | Red          |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | LightGBM + Sliding CQR       | 99.05%     |            7 |      0.2114 |       0.003536 |         0.893  |                 0.7137 | Green        |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | LightGBM + Online ACI        | 99.18%     |            6 |      0.218  |       0.003675 |         0.6027 |                 0.7533 | Green        |
| YESBANK.NS | yesbank_collapse_2018_2020 |    0.01 | LightGBM + Vol-Scaled ACI    | 99.05%     |            7 |      0.4477 |       0.005781 |         0.893  |                 0.7137 | Green        |

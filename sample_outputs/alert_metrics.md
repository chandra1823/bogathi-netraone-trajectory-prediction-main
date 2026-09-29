| Model | Level | Precision | Recall | F1 | False-alarm rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| KALMAN | window | 0.852 | 0.900 | 0.875 | 0.021 | 1693 | 294 | 188 | 13502 |
| KALMAN | track | 0.900 | 0.995 | 0.945 | 0.074 | 181 | 20 | 1 | 250 |
| LSTM | window | 0.891 | 0.915 | 0.903 | 0.015 | 1721 | 210 | 160 | 13586 |
| LSTM | track | 0.914 | 0.995 | 0.953 | 0.063 | 181 | 17 | 1 | 253 |

| Model | Mean lead time [s] | Median lead time [s] | Windows | Tracks |
|---|---|---|---|---|
| KALMAN | 4.88 | 4.72 | 15677 | 452 |
| LSTM | 4.56 | 4.40 | 15677 | 452 |

# Anomaly detector evaluation

How well does VoltEdge's anomaly detector actually work? This page measures it against known answers.

## Method

The data simulator injects unusual readings on purpose (a reading 2.5 to 4 times its normal level, on about 2% of hours) and labels each one. That gives a ground truth to score against.

- **Sites:** the four demo sites (factory, office, warehouse, data centre)
- **Data:** 90 days of hourly readings per site, repeated with 10 random seeds (21,600 hours per site)
- **Split:** train on the first 60 days, score the last 30 days, which the model has never seen. The split is by time, with no shuffling.
- **Scoring:** a planted anomaly counts as caught only if the detector flags that exact hour.

Run it yourself:

```bash
python3 scripts/evaluate_anomaly.py                     # current detector
python3 scripts/evaluate_anomaly.py --cleaned-readings  # previous behaviour, for comparison
```

## What the evaluation found

`EnergyTransformer` replaces extreme readings with interpolated values, so forecasts and charts are not thrown off by spikes. But the anomaly detector was trained and scored on those cleaned values, so the biggest spikes were smoothed away before the detector could see them.

The fix keeps the reading before cleaning in a new `kwh_raw` column. The anomaly detector now scores that real reading, while forecasting and charts still use the cleaned `kwh`.

## Results (all four sites, 10 seeds)

| | Caught | False alarms | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Before (cleaned readings) | 376 | 2,395 | 116 | 14% | 76% | 0.23 |
| After (raw readings) | 457 | 1,000 | 31 | 31% | 94% | 0.47 |

Recall rose from 76% to 94%, false alarms fell by 58%, and F1 roughly doubled.

### By site, after the fix

| Site | Caught | False alarms | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| LONDON-FACTORY-01 | 117 | 175 | 5 | 40% | 96% | 0.57 |
| DUBAI-OFFICE-01 | 97 | 175 | 25 | 36% | 80% | 0.49 |
| ROTTERDAM-WAREHOUSE-01 | 121 | 178 | 1 | 40% | 99% | 0.57 |
| FRANKFURT-DC-01 | 122 | 472 | 0 | 21% | 100% | 0.34 |

## Limits

- The data is simulated, so these numbers show the detector behaves as designed. They are not a guarantee on real meter data.
- Precision is held down partly by design: the hour after a spike also looks unusual (a sharp drop), and the detector flags around 5% of hours by default (`contamination=0.05`).
- The data centre has the most false alarms because its load is almost flat, so small changes look large once scaled. A per-site contamination setting would be the next thing to try.

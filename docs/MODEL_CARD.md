# Model card — ForecastGuard

**Model family:** LightGBM pooled quantile regression over five product-specific daily sales time series.  
**Task:** one-day-ahead recorded-sales quantile estimates and simplified inventory order-up-to recommendations.  
**Target population:** selected M5 `CA_1` food products; no claim that performance generalizes to all stores/items.

## Inputs

Product identifier; lagged realized sales from 1, 7, 14, and 28 days prior; trailing 7- and 28-day means; known weekday, month and event features. Input construction assumes yesterday's realized sales are known before today's prediction. Price availability is used for training-row filtering, **not** as a predictive feature.

## Outputs

- Q80: estimated conditional 80th percentile of observed unit sales; rounded up for an illustrative single-period order-up-to target using $2 surplus / $8 shortage costs.
- Q05 and Q95: nominal 90% prediction interval boundaries.
- Calibrated Q05/Q95: boundaries expanded using held-out, product-specific conformal nonconformity scores. Q80 remains unchanged.

## Training and evaluation

- Price-filtered eight-fold, expanding-window backtest: 2,240 product-day forecasts; relative simulated cost reduction of **21.64%** against an aligned seasonal quantile policy; 7/8 folds won.
- Raw vs conformalized nominal-90% interval historical coverage: **87.19%** vs **91.29%**; mean width **38.7** vs **44.7** units.
- Deployed historical demonstration: training through M5 day 1885, calibration days 1886–1941, inference on day 1942 (2016-05-23).

## Intended use

Teaching, research, comparative backtesting, and portfolio demonstration of uncertainty-aware forecasting and cost-sensitive decision support.

## Out-of-scope use

Production purchasing, stock replenishment, current Walmart forecasting, real profitability claims, and safety-critical ordering without availability data, live monitoring, cost validation, and prospective evaluation.

## Key caveats

1. Recorded sales can be censored by unavailable stock and do not equal true unconstrained demand.
2. Price records are only an imperfect proxy for product listing/availability.
3. Newsvendor costs are hypothetical and ignore inventory carryover, minimum order quantities, supplier lead time, and service constraints.
4. Quantile coverage, even when historically calibrated, may fail under temporal distribution shift or differ between products.
5. Historical periods were inspected during development; reported evaluation is not a blind holdout.
6. `joblib` bundles must be loaded only from trusted locations.

## Reproducibility

See repository README and `docs/TECHNICAL_REPORT.md`. Raw M5 data are not bundled with this public publication kit.

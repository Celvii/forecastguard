# Technical report — ForecastGuard

## Question
Can a pooled conditional demand-quantile model lower *simulated inventory decision costs* relative to a simple per-product seasonal quantile strategy, while reporting uncertainty with empirically evaluated coverage?

## Experimental setting

- M5 food sales for store CA_1, five top products selected using sales through the earliest fold's cutoff (day 1493).
- Daily observations from 2011–2016. The per-product working data contains 9,705 rows (5 × 1,941); feature construction removes the first 28 days of each series, leaving 9,565 rows. Of these, 336 observations lack sell-price records in the model-ready dataset.
- Numeric predictors: `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_mean_28`. Categorical predictors: product ID, weekday, month, and two known calendar-event names.
- No future realized sales are used in current-day feature construction; historical windows are aligned to daily sequential inference.
- Model: pooled LightGBM with quantile objective, α=0.8 for the reference inventory policy, 200 boosting rounds, learning rate 0.05, 15 leaves, minimum child samples 30, random seed 42.
- Comparator: empirical quantile calculated over eight preceding values from the corresponding weekday, per product.
- Eight expanding-window backtests, each 56 days, with cutoffs `[1493, 1549, 1605, 1661, 1717, 1773, 1829, 1885]`.

## Inventory decision objective

The simulated asymmetric cost for observed unit sales `y` and the suggested order-up-to amount `o` is:

`C(y, o) = c_over × max(o−y, 0) + c_under × max(y−o, 0)`

Under the simplified newsvendor assumptions, the decision quantile is `q = c_under/(c_under+c_over)`.

We use costs in hypothetical USD/unit, not actual Walmart accounting data. Orders are rounded up to whole units. We have no inventory on hand or physical demand censoring information.

## Core price-filtered rolling backtest

| Fold | Days evaluated | ML cost | Seasonal cost | Savings |
|---|---|---:|---:|---:|
| 1 | 1494–1549 | 12,108 | 14,292 | 15.28% |
| 2 | 1550–1605 | 10,930 | 16,696 | 34.54% |
| 3 | 1606–1661 | 11,104 | 14,286 | 22.27% |
| 4 | 1662–1717 | 10,306 | 10,304 | −0.02% |
| 5 | 1718–1773 | 12,948 | 16,118 | 19.67% |
| 6 | 1774–1829 | 9,840 | 16,472 | 40.26% |
| 7 | 1830–1885 | 10,994 | 13,130 | 16.27% |
| 8 | 1886–1941 | 9,746 | 10,980 | 11.24% |
| **Total** | **2,240 forecasts** | **87,976** | **112,278** | **21.64%** |

The 21.64% reduction is calculated using summed costs, **not** the arithmetic mean of fold percentages.

## Cost sensitivity

Three scenarios retrain the matching quantile model in each fold, comparing it to the corresponding seasonal quantile baseline.

| Scenario (surplus, shortage) | Target quantile | ML cost | Seasonal cost | Reduction | Winning folds |
|---|---:|---:|---:|---:|---:|
| $2, $2 | 0.50 | 45,152 | 65,212 | 30.76% | 8 / 8 |
| $2, $8 | 0.80 | 87,976 | 112,278 | 21.64% | 7 / 8 |
| $2, $18 | 0.90 | 118,980 | 156,200 | 23.83% | 7 / 8 |

## Data-quality investigation

Product `FOODS_3_090` has a 227-day recorded zero-sales streak with price records present throughout. Product `FOODS_3_120` has a 367-day streak; only 3 of those days have a price record. This motivates a price-filtered training sensitivity experiment, but neither case proves actual stockout conditions.

## Uncertainty evaluation

Separate LightGBM Q05/Q95 predictions are expanded using fold-specific, product-specific calibration margins on a chronologically disjoint calibration block. In a historical 2,240-forecast evaluation:

| Measure | Raw | Calibrated |
|---|---:|---:|
| Empirical interval coverage | 87.19% | 91.29% |
| Average width (units) | 38.7 | 44.7 |
| Calibrated interval score | — | approximately 59.4 |

We did not receive a sufficiently precise raw interval score in the conversation to report a meaningful numerical improvement; the dashboard displays an approximately zero difference. Wider intervals improve historical marginal coverage, but do not guarantee per-product conditional calibration or coverage under nonstationary demand.

## Deployment example

The calibrated next-day example is a forecast for historical day 1942 (2016-05-23), not a present-day Walmart forecast. The calibrated bundle trains through day 1885 and calibrates through day 1941. Q80 remains the decision target; only the Q05 and Q95 interval endpoints change with calibration.

## Threats to validity

Data leakage at product selection was audited: the top-five products selected using only days 1–1493 are identical to those selected through day 1773. However, model choices and reporting evolved after observing multiple validation windows. Thus these results are **not** a fully blind evaluation. Other limitations: censored sales, uncertain assortment/availability, few products, synthetic cost assumptions, missing inventory dynamics, and non-exchangeable time series.

## Data and licensing

The M5 competition dataset is identified by Kaggle as “Subject to Competition Rules.” See the original data page and review its terms before sharing underlying or row-level derived records. The public code-only repository does not require distributing those data files.

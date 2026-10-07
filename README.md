# Search Budget Optimizer

Python project that fits diminishing-returns response curves to campaign spend and conversion data, then reallocates a fixed budget across campaigns to maximise total conversions.

> **All data in this repository is simulated** (seeded random generator). No real company or campaign data is used. The simulated "current" allocation is a noisy, efficiency-aware human baseline, so the results illustrate the method on simulated data and are not a prediction for any real account.

## Method

1. **Simulate** 12 campaigns across Google Search, Bing Search, social display, video and display, with 52 weeks of spend and conversion data each. Conversions follow `a * spend^b` with log-normal noise (`0 < b < 1` gives diminishing returns).
2. **Fit** each campaign's response curve with log-log linear regression on the first 40 weeks.
3. **Validate** the fit on the last 12 held-out weeks (R² on training, MAPE on hold-out).
4. **Optimise** the allocation with SciPy SLSQP: maximise total conversions subject to a fixed total budget and per-campaign bounds of 70% to 150% of current spend.
5. **Evaluate** the new allocation against the true simulated curves (not just the fitted ones) and repeat over 200 random seeds to check robustness.

## Results (simulated data, weekly budget of about 107k units)

| Metric | Value |
|---|---|
| Campaigns | 12 |
| Average R² of fitted curves (training, log scale) | 0.83 |
| Average hold-out MAPE | 10.9% |
| Conversion uplift, seed 42 (true curves) | +14.3% |
| Mean uplift over 200 runs | +18.7% (median +18.4%) |
| Worst run / runs with positive uplift | +7.5% / 100% |

Outputs: `results_allocation.csv` (per-campaign current vs optimised spend), `results_summary.json`, `allocation_chart.png`, and `simulated_campaign_data.csv` (ready to load into Tableau).

## Run it

```bash
pip install -r requirements.txt
python optimizer.py
```

The chart (`allocation_chart.png`) opens automatically at the end of the run. Use `python optimizer.py --no-open` to skip that, for example on a server.

## Limitations

- The power-law response curve is a modelling assumption; real accounts show seasonality, auction effects, and interaction between campaigns.
- Spend bounds keep each change within a realistic range; the optimiser does not account for volume constraints on a given campaign.
- Next steps: test other curve shapes (e.g. Hill/saturation curves), add confidence intervals on the fitted parameters, and rerun on real, permitted data.

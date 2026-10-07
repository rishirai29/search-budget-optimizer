"""
Search Budget Optimizer
-----------------------
Fits diminishing-returns response curves to campaign spend/conversion data and
reallocates a fixed budget across campaigns to maximise total conversions.

IMPORTANT: all data here is SIMULATED (seeded random generator). No real
campaign or company data is used. The point of the project is the method:
response-curve fitting + constrained optimisation + validation.
The simulated "current" allocation is a noisy, efficiency-aware human baseline
(not a random or deliberately bad one), so uplift figures are illustrative of
the method on simulated data, not a prediction for any real account.

Model per campaign i:      conversions_i = a_i * spend_i ** b_i      (0 < b_i < 1)
Fit:                       log-log linear regression on training weeks
Optimise:                  maximise sum(a_i * s_i ** b_i)
                           subject to sum(s_i) = total budget,
                                      0.7 * current_i <= s_i <= 1.5 * current_i
Validation:                (1) hold-out weeks -> R^2 / MAPE of the fitted curves
                           (2) uplift evaluated on the TRUE simulated curves, not
                               just the fitted ones
                           (3) repeated over many random seeds
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMPAIGNS = [
    ("Google Search - Hotels TH", "Google Search"), ("Google Search - Hotels SG", "Google Search"),
    ("Google Search - Flights", "Google Search"), ("Google Search - Brand", "Google Search"),
    ("Bing Search - Hotels TH", "Bing Search"), ("Bing Search - Hotels SG", "Bing Search"),
    ("Bing Search - Flights", "Bing Search"), ("Bing Search - Brand", "Bing Search"),
    ("Meta Display - Prospecting", "Social Display"), ("Meta Display - Retargeting", "Social Display"),
    ("YouTube Video - Awareness", "Video"), ("Google Display - Remarketing", "Display"),
]
N_WEEKS, N_TRAIN = 52, 40
LOW, HIGH = 0.7, 1.5


def simulate(seed):
    rng = np.random.default_rng(seed)
    n = len(CAMPAIGNS)
    b_true = rng.uniform(0.45, 0.80, n)           # diminishing-returns exponent
    a_true = rng.uniform(1.0, 3.0, n)             # efficiency
    # "Current" allocation = a plausible human baseline: leans toward efficient
    # campaigns (a_i) but with substantial noise, i.e. good-but-imperfect judgement.
    base = 8000 * (a_true / a_true.mean()) ** 0.6 * rng.lognormal(0, 0.30, n)
    rows = []
    for i, (name, channel) in enumerate(CAMPAIGNS):
        spend = base[i] * rng.uniform(0.4, 1.8, N_WEEKS)
        conv = a_true[i] * spend ** b_true[i] * rng.lognormal(0, 0.12, N_WEEKS)
        for w in range(N_WEEKS):
            rows.append((name, channel, w + 1, spend[w], conv[w]))
    df = pd.DataFrame(rows, columns=["campaign", "channel", "week", "spend", "conversions"])
    return df, base, a_true, b_true


def fit_curves(df):
    out, metrics = [], []
    for name, g in df.groupby("campaign", sort=False):
        tr, te = g[g.week <= N_TRAIN], g[g.week > N_TRAIN]
        b, log_a = np.polyfit(np.log(tr.spend), np.log(tr.conversions), 1)
        a = np.exp(log_a)
        pred_tr = a * tr.spend ** b
        ss_res = ((np.log(tr.conversions) - np.log(pred_tr)) ** 2).sum()
        ss_tot = ((np.log(tr.conversions) - np.log(tr.conversions).mean()) ** 2).sum()
        mape = (np.abs(te.conversions - a * te.spend ** b) / te.conversions).mean()
        out.append((name, a, b))
        metrics.append((name, 1 - ss_res / ss_tot, mape))
    return pd.DataFrame(out, columns=["campaign", "a", "b"]), pd.DataFrame(metrics, columns=["campaign", "r2_train", "mape_holdout"])


def optimise(a, b, base):
    total = base.sum()
    f = lambda s: -np.sum(a * s ** b)
    grad = lambda s: -(a * b * s ** (b - 1))
    res = minimize(f, base.copy(), jac=grad, method="SLSQP",
                   bounds=list(zip(LOW * base, HIGH * base)),
                   constraints=[{"type": "eq", "fun": lambda s: s.sum() - total}],
                   options={"maxiter": 500, "ftol": 1e-9})
    return res.x


def run_once(seed):
    df, base, a_true, b_true = simulate(seed)
    fit, metrics = fit_curves(df)
    s_opt = optimise(fit.a.values, fit.b.values, base)
    conv_now_true = np.sum(a_true * base ** b_true)
    conv_opt_true = np.sum(a_true * s_opt ** b_true)
    conv_now_fit = np.sum(fit.a.values * base ** fit.b.values)
    conv_opt_fit = np.sum(fit.a.values * s_opt ** fit.b.values)
    return dict(df=df, base=base, s_opt=s_opt, fit=fit, metrics=metrics,
                uplift_true=conv_opt_true / conv_now_true - 1,
                uplift_fit=conv_opt_fit / conv_now_fit - 1,
                conv_now_true=conv_now_true, conv_opt_true=conv_opt_true)


if __name__ == "__main__":
    # --- primary run (seed 42): detailed outputs -------------------------------
    r = run_once(42)
    names = [c[0] for c in CAMPAIGNS]
    alloc = pd.DataFrame({"campaign": names, "current_weekly_spend": r["base"].round(0),
                          "optimised_weekly_spend": r["s_opt"].round(0)})
    alloc["change_pct"] = ((alloc.optimised_weekly_spend / alloc.current_weekly_spend - 1) * 100).round(1)
    alloc = alloc.merge(r["fit"].round(3), on="campaign").merge(r["metrics"].round(3), on="campaign")
    alloc.to_csv("results_allocation.csv", index=False)
    r["df"].to_csv("simulated_campaign_data.csv", index=False)

    # --- robustness: 200 seeds ---------------------------------------------------
    ups = np.array([run_once(s)["uplift_true"] for s in range(200)])
    r2 = r["metrics"].r2_train.mean()
    mape = r["metrics"].mape_holdout.mean()
    summary = dict(
        total_weekly_budget=float(r["base"].sum().round(0)),
        campaigns=len(CAMPAIGNS),
        seed42_uplift_on_true_curves_pct=round(r["uplift_true"] * 100, 2),
        seed42_uplift_on_fitted_curves_pct=round(r["uplift_fit"] * 100, 2),
        avg_r2_train_loglog=round(r2, 3),
        avg_mape_holdout_pct=round(mape * 100, 2),
        runs=200,
        mean_uplift_pct=round(ups.mean() * 100, 2),
        median_uplift_pct=round(np.median(ups) * 100, 2),
        min_uplift_pct=round(ups.min() * 100, 2),
        share_of_runs_positive_pct=round((ups > 0).mean() * 100, 1),
    )
    json.dump(summary, open("results_summary.json", "w"), indent=2)

    # --- chart -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(names)); h = 0.38
    ax.barh(y - h / 2, r["base"], h, label="Current allocation")
    ax.barh(y + h / 2, r["s_opt"], h, label="Optimised allocation")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Weekly spend (simulated currency units)")
    ax.set_title("Budget reallocation (same total budget) - simulated data")
    ax.legend(); plt.tight_layout(); plt.savefig("allocation_chart.png", dpi=150)

    print(json.dumps(summary, indent=2))
    print(alloc[["campaign", "current_weekly_spend", "optimised_weekly_spend", "change_pct"]].to_string(index=False))

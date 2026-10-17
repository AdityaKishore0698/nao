"""
analyze.py — group comparison as in Fears et al. (2023), section 2.6
====================================================================
Linear mixed-effects model on log-transformed DTW cost:

    log(DTW) ~ group * gesture * angle + child_arm + age,  random intercept per participant

followed by per gesture x angle group comparisons (the paper's Table 2).
Outliers >= 3 SD from the mean are removed first, as in the paper.
The paper used R/lme4 with Satterthwaite F-tests; statsmodels MixedLM gives
Wald tests, so p-values may differ slightly. Needs participants from both groups.

    python scripts/analyze.py --results data/results.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="data/results.csv")
    p.add_argument("--include-simulated", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["scored"].astype(str).str.lower() == "true"]
    if not args.include_simulated:
        df = df[df["source"] != "simulated"]
    df = df[df["group"].isin(["autistic", "neurotypical"])].copy()
    if df["group"].nunique() < 2 or df["participant"].nunique() < 4:
        raise SystemExit("Need scored trials from both groups (and several participants).")

    df["log_dtw"] = np.log(df["dtw_cost"])
    z = (df["log_dtw"] - df["log_dtw"].mean()) / df["log_dtw"].std()
    n_out = int((z.abs() >= 3).sum())
    df = df[z.abs() < 3]
    print(f"{len(df)} rows, {df['participant'].nunique()} participants, {n_out} outliers removed")

    has_age = df["age"].notna().all()
    formula = "log_dtw ~ C(group) * C(gesture) * C(angle) + C(child_arm)" + (" + age" if has_age else "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(formula, df, groups=df["participant"]).fit(reml=True)
    print(fit.summary())

    print("\nGroup comparison per gesture x angle (log DTW; autistic - neurotypical):")
    print(f"{'gesture':<10} {'angle':<18} {'autistic':>9} {'neurotyp':>9} {'diff':>7} {'p':>7}")
    for (g, a), sub in df.groupby(["gesture", "angle"]):
        if sub["group"].nunique() < 2 or sub["participant"].nunique() < 4:
            continue
        f = "log_dtw ~ C(group, Treatment('neurotypical')) + C(child_arm)" + (" + age" if has_age else "")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(f, sub, groups=sub["participant"]).fit(reml=True)
            term = [t for t in m.params.index if "autistic" in t][0]
            means = sub.groupby("group")["log_dtw"].mean()
            print(f"{g:<10} {a:<18} {means.get('autistic', np.nan):9.2f} "
                  f"{means.get('neurotypical', np.nan):9.2f} {m.params[term]:7.2f} {m.pvalues[term]:7.3f}")
        except Exception as exc:              # noqa: BLE001
            print(f"{g:<10} {a:<18} model failed: {exc}")


if __name__ == "__main__":
    main()

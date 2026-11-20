#!/usr/bin/env python3
"""Delete-one-PSU jackknife recomputation of the crude (M0) and primary (M2)
odds ratios, as described in the manuscript Methods. For each PSU, the PSU is
dropped and the weights of the remaining PSUs in its stratum are rescaled by
n_h / (n_h - 1); the model is refit and the jackknife variance is
sum_h [(n_h - 1) / n_h * sum_j (beta_hj - beta)^2]. CIs use t with 32 df.

Manuscript targets: M0 OR 1.62 (1.27-2.06); M2 OR 1.33 (1.03-1.72) by jackknife.
"""
import importlib.util
import numpy as np
import pandas as pd
from scipy import stats

spec = importlib.util.spec_from_file_location("analysis", "01_analysis.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

TCRIT = stats.t.ppf(0.975, 32)


def jackknife(d, outcome, terms, domain_mask, label):
    dd = d.copy()
    dd["__y__"] = dd[outcome]
    X, y, w, h, j, names = A.design_matrix(dd, terms, domain_mask)
    beta = A.fit_logit(X, y, w)
    k = names.index("noise")
    clusters = list(dict.fromkeys(f"{a}|{b}" for a, b in zip(h, j)))
    reps = []
    for c in clusters:
        ch, cj = c.split("|")
        in_stratum = h == float(ch)
        nh = len(set(j[in_stratum]))
        wr = w.copy()
        wr[in_stratum & (j == float(cj))] = 0.0
        wr[in_stratum & (j != float(cj))] *= nh / (nh - 1.0)
        reps.append((float(ch), nh, A.fit_logit(X, y, wr)[k]))
    var = 0.0
    for ch in sorted(set(r[0] for r in reps)):
        sub = [r for r in reps if r[0] == ch]
        nh = sub[0][1]
        var += (nh - 1.0) / nh * sum((r[2] - beta[k]) ** 2 for r in sub)
    se = np.sqrt(var)
    est = beta[k]
    print(f"{label:12s} jackknife OR {np.exp(est):.3f} "
          f"({np.exp(est - TCRIT * se):.3f}-{np.exp(est + TCRIT * se):.3f})")
    return np.exp(est), np.exp(est - TCRIT * se), np.exp(est + TCRIT * se)


def main():
    d = A.build()
    base = (d["RIDRETH1"] == 1) & d["age"].between(20, 69) & (d["RIDSTATR"] == 2)
    analytic = base & d["noise"].notna() & d["htn130"].notna()
    covs = ["age", "female", "edu", "born_outside", "bmi", "smoking",
            "diabetes", "offwork", "firearms", "cycle"]
    cc = analytic & d[covs].notna().all(axis=1)
    M2 = ["noise", "age", "female", "edu", "born_outside", "bmi", "smoking",
          "diabetes", "offwork", "firearms", "cycle"]
    rows = [("M0, crude",) + jackknife(d, "htn130", ["noise"], analytic, "M0, crude"),
            ("M2, primary",) + jackknife(d, "htn130", M2, cc, "M2, primary")]
    pd.DataFrame(rows, columns=["model", "OR_jackknife", "lo", "hi"]).to_csv(
        "results/table2_jackknife.csv", index=False)


if __name__ == "__main__":
    main()

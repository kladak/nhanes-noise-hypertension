#!/usr/bin/env python3
"""Reproduction analysis for: Self-reported occupational noise exposure and
hypertension among Mexican American adults aged 20 to 69 years (NHANES
2011-2012 and 2015-2016).

Implements the pipeline described in the manuscript Methods:
- Public-use files DEMO, AUQ, BPX, BPQ, BMX, SMQ, DIQ for both cycles.
- 4-year MEC weights (WTMEC2YR / 2); design specified on all MEC-examined
  respondents with a positive weight; study-population estimates are domain
  estimates (non-domain respondents contribute zero weight).
- Variances by Taylor series linearization; CIs use t with 32 design df
  (61 PSUs in 29 strata).
- Models: M0 crude; M1 age, sex, cycle; M2 fully adjusted (primary);
  sensitivity with the 140/90 mm Hg definition; sex-stratified M2;
  post hoc exposure x sex interaction.
- The delete-one-PSU jackknife recomputation lives in 03_jackknife.py.

Requires the XPT files in ./data (see download_nhanes.sh).
Outputs CSVs to ./results and prints a comparison against the values
reported in the manuscript.
"""
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data"
TCRIT = stats.t.ppf(0.975, 32)

FILES = ["DEMO", "AUQ", "BPX", "BPQ", "BMX", "SMQ", "DIQ"]


def load_cycle(suffix):
    frames = {}
    for f in FILES:
        df = pd.read_sas(f"{DATA}/{f}_{suffix}.XPT", format="xport")
        frames[f] = df.set_index("SEQN")
    out = frames["DEMO"].copy()
    for f in FILES[1:]:
        cols = [c for c in frames[f].columns if c not in out.columns]
        out = out.join(frames[f][cols], how="left")
    return out


def build():
    g = load_cycle("G")
    i = load_cycle("I")
    g["cycle"] = 0  # 2011-2012
    i["cycle"] = 1  # 2015-2016
    d = pd.concat([g, i], axis=0).reset_index(drop=True)

    # ---- exposure: AUQ330 (2011-2012) / AUQ331 (2015-2016), 1=Yes 2=No
    d["noise_raw"] = np.where(d["cycle"] == 0, d["AUQ330"], d["AUQ331"])
    d["noise"] = np.where(d["noise_raw"] == 1, 1.0,
                          np.where(d["noise_raw"] == 2, 0.0, np.nan))

    # ---- blood pressure: mean of readings 1-4; diastolic 0 set to missing
    sy = d[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]]
    di = d[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].replace(0, np.nan)
    d["sbp"] = sy.mean(axis=1, skipna=True)
    d["dbp"] = di.mean(axis=1, skipna=True)
    d["has_bp"] = sy.notna().any(axis=1) | di.notna().any(axis=1)

    def htn(data, s_cut, d_cut):
        meas = ((data["sbp"] >= s_cut) | (data["dbp"] >= d_cut)).where(data["has_bp"])
        dx = (data["BPQ020"] == 1)
        med = (data["BPQ050A"] == 1)
        out = np.where(dx | med, 1.0, meas)
        # usable status requires a BP reading or a positive diagnosis/medication
        # report; a "never told" answer with the medication item skipped and no
        # reading is missing (36 participants, per the manuscript flow)
        usable = data["has_bp"] | dx | med
        out = np.where(usable, out, np.nan)
        return pd.Series(out, index=data.index, dtype=float)

    d["htn130"] = htn(d, 130, 80)
    d["htn140"] = htn(d, 140, 90)

    # ---- covariates
    d["age"] = d["RIDAGEYR"]
    d["female"] = np.where(d["RIAGENDR"] == 2, 1.0,
                           np.where(d["RIAGENDR"] == 1, 0.0, np.nan))
    edu = d["DMDEDUC2"]
    d["edu"] = np.select([edu.isin([1, 2]), edu == 3, edu.isin([4, 5])],
                         [0.0, 1.0, 2.0], default=np.nan)  # <HS, HS, >HS
    d["born_outside"] = np.where(d["DMDBORN4"] == 2, 1.0,
                                 np.where(d["DMDBORN4"] == 1, 0.0, np.nan))
    d["bmi"] = d["BMXBMI"]
    smoker = np.full(len(d), np.nan)
    smoker = np.where(d["SMQ020"] == 2, 0.0, smoker)                       # never
    smoker = np.where((d["SMQ020"] == 1) & (d["SMQ040"] == 3), 1.0, smoker)  # former
    smoker = np.where((d["SMQ020"] == 1) & (d["SMQ040"].isin([1, 2])), 2.0, smoker)  # current
    d["smoking"] = smoker
    d["diabetes"] = np.where(d["DIQ010"] == 1, 1.0,
                             np.where(d["DIQ010"].isin([2, 3]), 0.0, np.nan))  # borderline = no
    # AUQ370 is the Yes/No item on 10+ h/week very loud noise or music off work
    d["offwork"] = np.where(d["AUQ370"] == 1, 1.0,
                            np.where(d["AUQ370"] == 2, 0.0, np.nan))
    d["firearms"] = np.where(d["AUQ300"] == 1, 1.0,
                             np.where(d["AUQ300"] == 2, 0.0, np.nan))

    d["wt"] = d["WTMEC2YR"] / 2.0
    d["design"] = (d["RIDSTATR"] == 2) & (d["WTMEC2YR"] > 0)
    return d


# ---------------------------------------------------------------- survey fit
def design_matrix(d, terms, domain):
    """Return X, y, w, strata, psu for the design frame with domain weights."""
    dd = d[d["design"]].copy()
    cols = [np.ones(len(dd))]
    names = ["(Intercept)"]
    for t in terms:
        if t in ("edu", "smoking"):
            for lev in (1.0, 2.0):
                cols.append((dd[t] == lev).astype(float).values)
                names.append(f"{t}{int(lev)}")
        elif t == "noise:female":
            cols.append((dd["noise"] * dd["female"]).values)
            names.append("noise:female")
        else:
            cols.append(dd[t].astype(float).values)
            names.append(t)
    X = np.column_stack(cols)
    y = dd["__y__"].values.astype(float)
    w = dd["wt"].values * domain[d["design"]].values.astype(float)
    # non-domain rows carry zero weight; sanitize their values so 0 * NaN
    # cannot poison the estimating equations
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    return X, y, w, dd["SDMVSTRA"].values, dd["SDMVPSU"].values, names


def fit_logit(X, y, w, tol=1e-10, maxit=100):
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (w * (y - p))
        W = w * p * (1.0 - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def taylor_var(X, y, w, beta, strata, psu):
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    W = w * p * (1.0 - p)
    H = (X * W[:, None]).T @ X
    Hinv = np.linalg.inv(H)
    score = X * (w * (y - p))[:, None]
    key = [f"{h}|{j}" for h, j in zip(strata, psu)]
    totals = {}
    for k in dict.fromkeys(key):
        mask = np.array([kk == k for kk in key])
        totals[k] = score[mask].sum(axis=0)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for h in pd.unique(strata):
        js = [k for k in totals if k.split("|")[0] == str(h)]
        nh = len(js)
        if nh < 2:
            continue
        arr = np.stack([totals[k] for k in js])
        cen = arr - arr.mean(axis=0)
        meat += (nh / (nh - 1.0)) * (cen.T @ cen)
    return Hinv @ meat @ Hinv


def run_model(d, outcome, terms, domain_mask, label):
    dd = d.copy()
    dd["__y__"] = dd[outcome]
    X, y, w, h, j, names = design_matrix(dd, terms, domain_mask)
    beta = fit_logit(X, y, w)
    V = taylor_var(X, y, w, beta, h, j)
    k = names.index("noise")
    est, se = beta[k], np.sqrt(V[k, k])
    orr = np.exp(est)
    lo, hi = np.exp(est - TCRIT * se), np.exp(est + TCRIT * se)
    p = 2 * stats.t.sf(abs(est / se), 32)
    n = int(domain_mask[d["design"]].sum())
    print(f"{label:44s} n={n:5d}  OR {orr:.3f} ({lo:.3f}-{hi:.3f})  p={p:.4f}")
    return dict(label=label, n=n, OR=orr, lo=lo, hi=hi, p=p, beta=beta, V=V, names=names)


def wtd_mean(d, var, domain_mask):
    dd = d[d["design"]]
    w = dd["wt"].values * domain_mask[d["design"]].values.astype(float)
    x = dd[var].values.astype(float)
    ok = ~np.isnan(x)
    w, x = w * ok, np.where(ok, x, 0.0)
    est = (w * x).sum() / w.sum()
    resid = w * (x - est) / w.sum()
    key = [f"{h}|{j}" for h, j in zip(dd["SDMVSTRA"].values, dd["SDMVPSU"].values)]
    totals = {}
    for k in dict.fromkeys(key):
        mask = np.array([kk == k for kk in key])
        totals[k] = resid[mask].sum()
    var_est = 0.0
    for h in pd.unique(dd["SDMVSTRA"].values):
        js = [k for k in totals if k.split("|")[0] == str(h)]
        nh = len(js)
        if nh < 2:
            continue
        arr = np.array([totals[k] for k in js])
        var_est += (nh / (nh - 1.0)) * ((arr - arr.mean()) ** 2).sum()
    return est, np.sqrt(var_est)


def main():
    d = build()
    des = d[d["design"]]
    n_psu = des.groupby(["SDMVSTRA", "SDMVPSU"]).ngroups
    n_str = des["SDMVSTRA"].nunique()
    print(f"design: {n_psu} PSUs in {n_str} strata, df = {n_psu - n_str}")

    flow = {}
    flow["Mexican American participants (both cycles)"] = int((d["RIDRETH1"] == 1).sum())
    ma = (d["RIDRETH1"] == 1)
    flow["Aged 20 to 69"] = int((ma & d["age"].between(20, 69)).sum())
    flow["Examined at MEC"] = int((ma & d["age"].between(20, 69) & (d["RIDSTATR"] == 2)).sum())
    base = ma & d["age"].between(20, 69) & (d["RIDSTATR"] == 2)
    flow["Answered noise item Yes/No"] = int((base & d["noise"].notna()).sum())
    analytic = base & d["noise"].notna() & d["htn130"].notna()
    flow["Analytic sample"] = int(analytic.sum())
    covs = ["age", "female", "edu", "born_outside", "bmi", "smoking",
            "diabetes", "offwork", "firearms", "cycle"]
    cc = analytic & d[covs].notna().all(axis=1)
    flow["Adjusted models (complete cases)"] = int(cc.sum())
    for k, v in flow.items():
        print(f"  flow: {k:45s} {v}")
    pd.DataFrame({"step": list(flow), "n": list(flow.values())}).to_csv(
        "results/sample_flow.csv", index=False)

    prev, se = wtd_mean(d, "noise", analytic)
    print(f"exposure prevalence: {100*prev:.1f}% (SE {100*se:.1f})")
    for lab, m in [("exposed", analytic & (d["noise"] == 1)),
                   ("unexposed", analytic & (d["noise"] == 0))]:
        for oc in ("htn130", "htn140"):
            p_, s_ = wtd_mean(d, oc, m)
            print(f"  {oc} prevalence {lab}: {100*p_:.1f}% "
                  f"({100*(p_-TCRIT*s_):.1f}-{100*(p_+TCRIT*s_):.1f})")

    M2 = ["noise", "age", "female", "edu", "born_outside", "bmi", "smoking",
          "diabetes", "offwork", "firearms", "cycle"]
    rows = []
    rows.append(run_model(d, "htn130", ["noise"], analytic, "M0, crude"))
    rows.append(run_model(d, "htn130", ["noise", "age", "female", "cycle"], cc,
                          "M1, age, sex, and cycle"))
    rows.append(run_model(d, "htn130", M2, cc, "M2, fully adjusted (primary)"))
    rows.append(run_model(d, "htn140", M2, cc, "Sensitivity: 140/90, fully adjusted"))
    rows.append(run_model(d, "htn130", [t for t in M2 if t != "female"],
                          cc & (d["female"] == 0), "Men, fully adjusted"))
    rows.append(run_model(d, "htn130", [t for t in M2 if t != "female"],
                          cc & (d["female"] == 1), "Women, fully adjusted"))
    inter = run_model(d, "htn130", M2 + ["noise:female"], cc, "M2 + interaction (noise coef)")
    k = inter["names"].index("noise:female")
    b, s = inter["beta"][k], np.sqrt(inter["V"][k, k])
    print(f"  interaction ratio of ORs: {np.exp(b):.3f} "
          f"({np.exp(b-TCRIT*s):.3f}-{np.exp(b+TCRIT*s):.3f})  "
          f"p={2*stats.t.sf(abs(b/s),32):.3f}")
    pd.DataFrame(rows)[["label", "n", "OR", "lo", "hi", "p"]].to_csv(
        "results/table2_models.csv", index=False)

    # ---- Table 1: weighted characteristics by exposure, estimate (SE)
    for name, (srcv, lv) in {"edu0": ("edu", 0), "edu1": ("edu", 1), "edu2": ("edu", 2),
                             "smk0": ("smoking", 0), "smk1": ("smoking", 1),
                             "smk2": ("smoking", 2)}.items():
        d[name] = np.where(d[srcv].isna(), np.nan, (d[srcv] == lv) * 100.0)
    # (label, column, scale): scale 100 turns a 0/1 proportion into a percent
    specs = [("Age, y, mean (SE)", "age", 1), ("Female, % (SE)", "female", 100),
             ("Education: less than high school, % (SE)", "edu0", 1),
             ("Education: high school or equivalent, % (SE)", "edu1", 1),
             ("Education: more than high school, % (SE)", "edu2", 1),
             ("Born outside the US, % (SE)", "born_outside", 100),
             ("Body mass index, kg/m2, mean (SE)", "bmi", 1),
             ("Smoking: never, % (SE)", "smk0", 1),
             ("Smoking: former, % (SE)", "smk1", 1),
             ("Smoking: current, % (SE)", "smk2", 1),
             ("Diabetes (self-reported), % (SE)", "diabetes", 100),
             ("Off-work very loud noise, % (SE)", "offwork", 100),
             ("Ever used firearms, % (SE)", "firearms", 100),
             ("Systolic BP, mm Hg, mean (SE)", "sbp", 1),
             ("Diastolic BP, mm Hg, mean (SE)", "dbp", 1),
             ("Hypertension, 130/80 definition, % (SE)", "htn130", 100),
             ("Hypertension, 140/90 definition, % (SE)", "htn140", 100),
             ("2015-2016 cycle, % (SE)", "cycle", 100)]
    groups = [("Overall (n = 1,258)", analytic),
              ("Occupational noise, yes (n = 481)", analytic & (d["noise"] == 1)),
              ("Occupational noise, no (n = 777)", analytic & (d["noise"] == 0))]
    t1 = []
    for label, var, scale in specs:
        row = {"Characteristic": label}
        for gname, gm in groups:
            est, se = wtd_mean(d, var, gm)
            row[gname] = f"{est * scale:.1f} ({se * scale:.1f})"
        t1.append(row)
    pd.DataFrame(t1).to_csv("results/table1.csv", index=False)
    print("wrote results/table1.csv")


if __name__ == "__main__":
    main()

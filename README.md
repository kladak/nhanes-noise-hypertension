# Occupational noise exposure and hypertension in Mexican American adults: NHANES 2011-2012 and 2015-2016

Analysis code for the manuscript:

> Karim Ladak. *Self-reported occupational noise exposure and hypertension among Mexican American adults aged 20 to 69 years: a cross-sectional analysis of NHANES 2011-2012 and 2015-2016.*

This repository contains the code that reproduces every value reported in the manuscript (sample flow, Table 1, Table 2, and the jackknife recomputation) from the public-use NHANES files. No NHANES data are redistributed here; the data are downloaded from the National Center for Health Statistics (NCHS).

## Data

Download the public-use files from NCHS (https://wwwn.cdc.gov/nchs/nhanes/):

```bash
bash download_nhanes.sh
```

This fetches the DEMO, AUQ, BPX, BPQ, BMX, SMQ and DIQ files for 2011-2012 (suffix `_G`) and 2015-2016 (suffix `_I`) into `data/`, using the URL pattern `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/<FILE>_<suffix>.XPT`.

## Run

```bash
pip install -r requirements.txt
python 01_analysis.py     # sample flow, prevalences, Table 1, Table 2 (Taylor linearization)
python 03_jackknife.py    # delete-one-PSU jackknife recomputation of M0 and M2
```

Outputs are written to `results/` (`sample_flow.csv`, `table1.csv`, `table2_models.csv`, `table2_jackknife.csv`); the copies in this repository are the outputs of a verified run.

## Methods in brief

- Two cycles combined with 4-year MEC weights (WTMEC2YR / 2). The survey design (masked strata SDMVSTRA, PSUs SDMVPSU) is specified on all MEC-examined respondents with a positive weight (61 PSUs in 29 strata, 32 design degrees of freedom); study-population estimates are domain estimates.
- Variances by Taylor series linearization; confidence intervals use the t distribution with 32 df. The jackknife script provides the independent delete-one-PSU variance check.
- Exposure: AUQ330 (2011-2012) / AUQ331 (2015-2016), ever a job with loud noise 4+ hours a day, several days a week. Outcome: mean measured BP >= 130/80 mm Hg (2017 ACC/AHA), prior diagnosis, or current antihypertensive medication; sensitivity analysis at 140/90 mm Hg.
- Models: M0 crude; M1 age, sex, cycle; M2 fully adjusted (age, sex, education, nativity, BMI, smoking, diabetes, off-work noise, firearm use, cycle); sex-stratified M2; post hoc exposure x sex interaction.

## Verified reproduction

Running the scripts on the public files reproduces, to the rounding used in the manuscript: sample flow 3,276 / 1,372 / 1,319 / 1,294 / 1,258 / 1,240; exposure prevalence 37.8%; hypertension prevalence 46.2% (exposed) vs 34.7% (unexposed); ORs M0 1.62 (1.27-2.06), M1 1.44 (1.13-1.84), M2 1.33 (1.04-1.72), 140/90 sensitivity 1.65 (1.17-2.34), men 1.11 (0.70-1.75), women 1.74 (1.07-2.82), interaction ratio of ORs 1.52 (0.70-3.32); jackknife M0 1.616 (1.267-2.062) and M2 1.334 (1.034-1.721).

## Dependencies

Python 3 with pandas, NumPy and SciPy (see `requirements.txt`).

"""WEEE data-quality evidence tables reported in the Supplement.

1. Body-mass evidence: athlete profile recorded by the VO2 Master app (Details.txt) vs Demographics.csv, BMI check,
   app mass implied by VO2[mL/min] / VO2[mL/kg/min], and the resulting target factor.
2. HR-screen statistics (target-free): per-condition median Wahoo HR, exercise-minus-rest rise, Wahoo–Zephyr gap, pass.
Output: results_r1/K100/WEEE/analysis/weee_body_mass_evidence.csv, weee_hr_screen.csv"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import data_external as X  # noqa: E402
from r1.datasets import hr_plausible  # noqa: E402

OUT = ROOT / "results_r1" / "K100" / "WEEE" / "analysis"


def body_mass():
    dem = pd.read_csv(X.WEEE_ROOT / "Demographics.csv").set_index("Participant")
    rows = []
    for p in dem.index:
        det = sorted((X.WEEE_ROOT / p).rglob("Details.txt"))
        avg = X.WEEE_ROOT / p / "VO2" / "DataAverage.csv"
        if not det or not avg.exists():
            continue
        t = det[0].read_text(errors="ignore")
        prof = re.search(r"Athlete Name:\s*(\S+)", t).group(1)
        m_prof = float(re.search(r"Weight:\s*([\d.]+)", t).group(1))
        age_prof = int(re.search(r"Age:\s*(\d+)", t).group(1))
        a = pd.read_csv(avg, usecols=["VO2[mL/min]", "VO2[mL/kg/min]"])
        a = a[(a["VO2[mL/min]"] > 0) & (a["VO2[mL/kg/min]"] > 0)]
        h = dem.loc[p, "Height"] / 100
        rows.append({"participant": p, "app_profile": prof, "profile_age": age_prof, "demographics_age": int(dem.loc[p, "Age"]),
                     "profile_mass_kg": m_prof, "app_implied_mass_kg": round(float((a["VO2[mL/min]"] / a["VO2[mL/kg/min]"]).median()), 1),
                     "demographics_mass_kg": float(dem.loc[p, "Weight"]), "listed_BMI": float(dem.loc[p, "BMI"]),
                     "BMI_from_demographics_mass": round(dem.loc[p, "Weight"] / h ** 2, 1), "BMI_from_profile_mass": round(m_prof / h ** 2, 1),
                     "target_factor_corrected_over_app": round(m_prof / dem.loc[p, "Weight"], 3)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "weee_body_mass_evidence.csv", index=False)
    return t


def hr_screen():
    df = X.load_weee("protocol").df
    med = df.groupby(["pid", "condition"])["HR"].median().unstack()
    medz = df.groupby(["pid", "condition"])["HR_zephyr"].median().unstack()
    ex = med[[c for c in med.columns if c.startswith(("Cycle", "Run"))]].max(axis=1)
    rest = med[[c for c in med.columns if c in ("Sit", "Stand")]].max(axis=1)
    t = med.add_prefix("wahoo_median_").join(medz.add_prefix("zephyr_median_"))
    t["exercise_minus_rest_bpm"] = ex - rest
    t["wahoo_zephyr_gap_bpm"] = (med - medz).abs().median(axis=1)
    t["passes_screen"] = pd.Series(hr_plausible(df))
    t.round(1).to_csv(OUT / "weee_hr_screen.csv")
    return t


if __name__ == "__main__":
    print(body_mass().to_string(index=False))
    print(hr_screen()[["exercise_minus_rest_bpm", "wahoo_zephyr_gap_bpm", "passes_screen"]].round(1).to_string())

# External datasets

The two external datasets are not redistributed here. Download them from their repositories and place them as
shown below (or set the environment variable `VO2_EXTERNAL_DATA` to a folder with the same layout).

```
data/external/
  weee_raw/extracted/dataset/     WEEE: contents of dataset.zip (P01 ... P17, Demographics.csv, Study_Information.csv)
  ingraham_raw/                   Ingraham: Subject01.mat ... Subject10.mat
  weee_clock_offsets.csv          included: Zephyr clock offsets (regenerate: scripts_r1/make_weee_offsets.py)
  weee_inclusion_v04.csv          included: data-quality inclusion list, P06 and P14 excluded (check: scripts_r1/make_weee_inclusion.py)
  cache/                          created automatically (processed 1 Hz panels)
```

| Dataset | Source | Notes |
|---|---|---|
| WEEE | Gashi et al. (2022) *Sci. Data* 9:537, https://doi.org/10.1038/s41597-022-01643-5; data: Zenodo, https://doi.org/10.5281/zenodo.6420886 (`dataset.zip`, md5 `23e411c566f1c734e74e23fa76bd1ab0`) | CC BY 4.0. V̇O₂ is taken from the analyser's 1 Hz `VO2[mL/min]` output and divided by the body mass in `Demographics.csv`. |
| Ingraham | Ingraham, Ferris & Remy (2019) *J. Appl. Physiol.* 126:717–729, https://doi.org/10.1152/japplphysiol.00714.2018; data: figshare, https://doi.org/10.6084/m9.figshare.7473191 | `Subject01.mat` … `Subject10.mat`. Breath-by-breath V̇O₂ is interpolated to 1 Hz. |

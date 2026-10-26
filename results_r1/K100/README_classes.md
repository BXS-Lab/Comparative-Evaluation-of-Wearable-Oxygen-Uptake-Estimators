# Class codes in the result files

`table5.csv`, `replication.csv` and the `*_class` columns of the analysis files use internal class codes. The paper
classifies every family contrast with one rule, implemented in `scripts_r1/figstyle.py:contrast_label`:

- **Equivalent** if Δ* < 0.05 × SD of V̇O₂ (equivalence margin); *borderline* if Δ* > 0.95 × margin;
  if the difference is also resolved, the direction is reported ("X better within margin").
- otherwise **X better** (resolved difference) if p_Holm < 0.05 and the 95% CI excludes zero;
- otherwise **Not resolved**.

0.10 × SD (`m_lenient`) is a reference value only.

| Code in result files | Label in the paper |
|---|---|
| `O1` | Equivalent (Δ* < 0.05 SD) |
| `O1L` | Not resolved under the rule above |
| `O2-Tree`, `O2-Temporal`, `ML better` | Tree / Temporal better (resolved difference) |
| `O3`, `unresolved` | Not resolved |

| Replication code | Wording in the paper |
|---|---|
| `replicated` | replicated |
| `contradicted` | not replicated |
| `not resolved externally` | not resolved externally |
| `BEL unresolved` | not resolved on the primary dataset |

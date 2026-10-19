# Numerical sources and the author-approved acceptance rule

Paper-reference tables and archived predictions remain unchanged. The saved
checkpoint-evaluation path is a separate source and can differ slightly.
The author-approved release tolerance is: no reversal of a compared ranking,
OR absolute mean difference no greater than 1 percentage point for percentage
metrics (0.01 for fraction-scale metrics). This is a release acceptance rule,
not a statistical significance test or a claim of exact equality.

The seven previously audited reference tables contain 216 comparison rows
(including repeated measures across main and supplementary tables). All pass
this rule; the largest observed percentage-metric mean difference is
0.13913737311625596 percentage points. Archived predictions independently
reproduce all seven reference tables at the original rounding precision.
These checks do not cover every table, all fits, or fresh GPU inference.

Keep archived predictions and checkpoint-evaluation results distinct. Neither
has been overwritten, rounded to hide differences, or substituted silently.

All audited rows, exact-match diagnostics, units, mean differences and author-rule decisions are available in [checkpoint_comparison.csv](checkpoint_comparison.csv).

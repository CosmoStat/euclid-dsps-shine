# Completed exploratory blind parent, 2026-09-27

Run: `avi_reference_to_parent_20260927_184817`.
Contract: `560505c9caeee1702512cade7fee585a43c50475b02fd66dcfefe6b4910310f3`.
User SLURM output: population 252749 and report 252750 COMPLETED.
Local readback: 23 receipt artifacts verify; `reference/basis.npz` and
`population/best.eqx` were excluded from transfer. Simulation banks are not
local. Do not mistake small-artifact verification for full bank/model validation.

## Progress and remaining discrepancy

The recovery succeeds without loosening the KKT tolerance. All four regularized
fits have KKT gaps below 2e-6; selected penalty is 0.003. Parent and selected
weights sum to one; saved u exactly matches normalized v/alpha. Component alpha
ranges from 0.09399 to 1, so this fit is not dividing by near-zero efficiencies.
Learned alpha is 0.60235 versus validation 0.60995, an absolute difference of
0.00760. Weak-parent support constraint is inactive.

Inverse squared weight effective component count is 19.28/256, maximum weight
0.11237, top three mass 0.32064. This concentration alone is not collapse or an
identifiability certificate: the successful physical-capacity LP is sparse too.

| Physical parameter | Parent W1/IQR | Selected W1/IQR | Parent capacity control |
|---|---:|---:|---:|
| Redshift | 0.07744 | 0.04178 | 0.03419 |
| Log stellar mass | 0.05272 | 0.03071 | 0.05277 |
| Log metallicity | 0.04037 | 0.02863 | 0.01275 |
| Dust Av | 0.18677 | 0.17356 | 0.01933 |
| Dust slope | 0.01719 | 0.02011 | 0.02852 |

Only Av fails the 0.1 physical marginal gate. The learned curve has too much
weight at low Av and a redshift shape discrepancy. These are normalized
distribution distances, not fractional errors in individual galaxy estimates.
The physical joint SW is 0.09751 parent and 0.07666 selected, above the 0.05
gate; capacity control is 0.02798 parent and 0.04904 selected. Thus useful
physical family capacity exists, but blind fitting does not recover it here.
The selected mismatch already exists before parent inverse-selection correction.

The previous coherent blind fit reported parent SW 0.12394 and selected SW
0.11820. New scores are lower, but different draws/evaluation splits and a new
family prevent treating this as a controlled percentage improvement. Current
within-run controls are more informative than a cross-run percentage.

## Classifier and predictive qualifications

Best validation is epoch 399, NLL 1.59957; stopping is epoch 460 after no qualifying
improvement. Independent audit NLL 1.60638 versus null 5.35584 confirms useful
information. This is the implemented plateau criterion, not proof of global
neural optimization or sufficient conditional ratio accuracy.

Independent ratio-moment error is 0.03170 versus gate 0.030. It estimates a
relative component-frequency identity on 68,594 rows, after offset calibration
on a different 68,250 rows. No sampling uncertainty is saved. This slight failure
cannot by itself be blamed for the much larger dust error; passing the identity
would not certify the conditional ratios either.

Learned flux marginal KS ranges 0.00854 to 0.02220, below 0.05. Mass above each
target q99.9 is 0.139%-0.160% versus target 0.1066%, below gate 0.5%.
Those PASS labels concern marginal CDFs and tail frequency, NOT joint colors or
tail amplitudes. Learned r-band raw W1/scale is still 4.489 and becomes 0.221
under diagnostic upper winsorization. The scale is max(target IQR, 1e-30), not
always the literal IQR. No observations, weights or simulations are clipped.

Crucially, the truth-assisted physical capacity control has about 1.43% beyond
the same brightness threshold, roughly 13.4 times the target fraction. Its raw
r-band W1/scale is 37949.6. It fits physical distributions better but does not
provide a validated 15D photometric population model. Both control and blind
fit retain SFH SW about 0.38 and substantial late-contrast marginal errors.

Interpretation: the family/conditional SFH, rare native anchors, kernels,
classifier errors and physical degeneracies remain candidate causes. The
physical control was not optimized for fluxes, so its failure does NOT prove
no weights in this family could jointly fit physical and observable targets.
Underconstrained individual SFH is acceptable; an incorrect distribution of SFH
in forward simulations can nonetheless bias the physical population. No causal
SFH-to-dust attribution or selection-algebra bug is demonstrated by these files.

## One next experiment, not another large run

Recommended in this review, now implemented in the
[recovery check runbook](feniks_parent_recovery.md), not remotely submitted: one frozen-model
in-family recovery check using the existing reserved bank. No target resimulation,
DSPS, new classifier training or final posterior. Use two declared known parent
weight vectors (saved blind and capacity-control weights), and independent bank
subsets for pseudo-catalogue fitting and density evaluation. Preserve all 15D
coordinates. Record finite-bank effective sizes and conditional label counts.

1. Infer selected/parent weights from those pseudo-observed fluxes with the
   existing frozen classifier, offsets and fixed scientific fitting rule.
2. Compare recovered parent/selected physical distributions, especially Av,
   with the KNOWN generating mixtures, not the coherent catalogue truth.
3. Use saved true component labels as an upper-bound control for sampling and
   selection bookkeeping. Separate finite-sample/prior-penalty displacement
   from approximate-ratio error; do not make exact 256-weight L1 recovery a gate
   for overlapping components.
4. From the same reserved rows, attribute bright-tail probability AND excess
   flux to component/physical/SFH groups. Joint flux/color closure matters too.
   No expensive new multi-seed/bootstrap sweep; report uncertainty for the
   near-threshold ratio moment rather than interpreting a naked PASS/FAIL.

If recovery fails even inside the family, fix ratio calibration/estimation or
regularization before generating a new bank. If it succeeds but the coherent
catalogue remains biased, prioritize joint reference/conditional support and
degeneracy diagnosis. It does not automatically prove which SFH correction is
right. In either case, a final parent-trained posterior would inherit current
parent error and is not the next justified large compute expense.

The existing strict/--resume/--explore-cdf launchers do not perform this new
diagnostic. Do not resubmit them unchanged expecting a different scientific test.
Use `submit_feniks_parent_recovery.sh` from the new runbook. The implementation
fixes the source-selected penalty (no sweep), uses all selected role-4 rows with
stratified importance weights and evaluates on disjoint role 3.

## Where to look

Under the synchronized run directory:

- `analysis/diagnosis_summary.png`: marginal error localization, joint distances,
  classifier history and bright-tail counts in one figure.
- `report/parent_physical.png`: blue truth, orange blind estimate, green physical
  control. Look first at Av, then redshift; mass and metallicity are not lost.
- `report/selected_physical.png`: the same Av bias exists in the selected sample.
- `analysis/sfh_and_flux_tails.png`: unresolved SFH and tail amplitude despite
  passing the tail-frequency gate.
- `report/observable_predictive.png`: good bulk CDFs are not joint closure.
- `analysis/SUMMARY.json`: verified files, normalization and precise metrics.

Reproduce local review without banks:

```bash
python -m scripts.analyze_feniks_reference_to_parent \
  outputs/forward_population_results/avi_reference_to_parent_20260927_184817
```

No thresholds, source artifacts, target catalogue or fitted prior were changed.
Final independent posterior calibration and publication validation remain open.

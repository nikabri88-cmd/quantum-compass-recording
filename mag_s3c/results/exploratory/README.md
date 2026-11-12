# MAG-S3c: exploratory computations after the confirmatory run

Computed on 11 November 2026 (UTC), after the run and its reanalysis, with the frozen functions of code/mag_s3c.py; nothing here changes a verdict. The script is explore_after_run.py, and its outputs are the JSON and log files of this folder.

## 1. The scale of CISS of Kobori et al. (2026)

Kobori et al. (2026) estimated the contribution of CISS in the cryptochrome-DASH of the frog Xenopus laevis by time-resolved EPR and a model fit at about 8% for the pair FAD–W_B and at most 6% for FAD–W_C (Kobe University press release); carrying these numbers over to Cry4 and to W_D is an assumption. Form K at m = 1 (CISS at birth only, full non-reciprocity) and the control P-0 (the polarised birth state of the polarisation forms, recombining through the singlet) follow exactly from the reference solutions of the run for any CISS angle, so no new solves are needed. The script rebuilds both at the angles that the two readings of the estimate give, sin χ = 0.06 or 0.08 (χ = 0.0191π or 0.0255π) and sin²χ = 0.06 or 0.08 (χ = 0.0788π or 0.0913π), together with χ = 0.1π, where every value equals the run (check rows). Output: scale.json, scale.log.

Form K, CISS axis r̂, the stricter of the two transfers at p = 1 and S = 1 (here always F_amp); threshold of the dipole test 0.01:

| Pair and level | χ = 0.1π (run) | sin χ = 0.06 | sin χ = 0.08 | sin²χ = 0.06 | sin²χ = 0.08 |
|---|---|---|---|---|---|
| W_C, no relaxation | 0.0578 | 0.0117 | 0.0156 | 0.0466 | 0.0533 |
| W_C, random fields 10⁶ s⁻¹ | 0.0505 | 0.0102 | 0.0136 | 0.0407 | 0.0466 |
| W_C, dephasing 10⁶ s⁻¹ | 0.0604 | 0.0122 | 0.0163 | 0.0487 | 0.0557 |
| W_C, dephasing 3·10⁷ s⁻¹ | 0.0478 | 0.0097 | 0.0129 | 0.0385 | 0.0441 |
| W_D, no relaxation | 0.1804 | 0.0367 | 0.0489 | 0.1458 | 0.1666 |
| W_D, random fields 10⁶ s⁻¹ | 0.1530 | 0.0311 | 0.0414 | 0.1236 | 0.1412 |
| W_D, dephasing 10⁶ s⁻¹ | 0.1845 | 0.0376 | 0.0500 | 0.1491 | 0.1703 |
| W_D, dephasing 3·10⁷ s⁻¹ | 0.0965 | 0.0198 | 0.0263 | 0.0781 | 0.0892 |

The response of form K grows almost exactly as sin χ cos χ, the coherence term of its exact split, so the amplitude reading moves W_C to the threshold of the dipole test at p = 1 and S = 1 even with full non-reciprocity, while W_D stays 2–4 times above it; axial disorder multiplies the response by g(S), 1.66 at S = 0.4. The triplet-share reading keeps the margins close to those at 0.1π. P-0 at sin χ = 0.06 gives 0.030–0.042 in W_C and 0.036–0.052 in W_D for the axis z without relaxation and at 10⁶ s⁻¹, and 0.005 at dephasing 3·10⁷ s⁻¹ (scale.json).

## 2. H3 in the trial run of 10 November

The sealed line of the trial run (substitute rates, 60 directions) had H3 not met in W_C and W_D, and the run has it met. On 60 directions the harmonic fit of degree 10 is underdetermined: 66 even and 55 odd coefficients against 30 antipodal pairs, and least squares returns the solution of smallest norm. The script solves the reference anew with the frozen solver on the grid of 60 directions and rebuilds form K at χ_b = 0.1π, m = 1, axis r̂, on the levels of H3.

| Grid and rates | W_C: F | W_C: F_amp | W_D: F | W_D: F_amp |
|---|---|---|---|---|
| 2000 directions, Cry4a rates (the run) | 0.47–0.71 | 0.050–0.060 | 3.9–4.9 | 0.153–0.184 |
| 60 directions, Cry4a rates | 0.0001 | 0.037–0.045 | 0.0002–0.0003 | 0.115–0.139 |
| 60 directions, substitute rates (the trial) | 0.0001 | 0.053–0.062 | 0.0003–0.0004 | 0.155–0.186 |
| 2000 directions, substitute rates | 0.61–0.87 | 0.072–0.084 | 3.2–3.8 | 0.207–0.248 |

On 60 directions the even part spreads over the degrees the grid cannot resolve, the even range of the receptor grows, and the ratio transfer F falls to about 10⁻⁴, below the threshold; the amplitude transfer, which takes only the dipole, keeps 74–76% of its value on 2000 directions. On 2000 directions the substitute rates raise F_amp by about 40%, and H3 holds there with a margin of 7–8 times in W_C and 21–25 times in W_D. The grid of 60 directions thus explains why H3 failed in the trial: with the frozen code, the same grid and rates give H3 not met. The trial itself ran on code from before the freeze, so this recomputation is no replay of it.

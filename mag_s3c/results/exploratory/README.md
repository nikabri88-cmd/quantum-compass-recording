# MAG-S3c: exploratory computations after the confirmatory run

Computed on 11 November 2026 (UTC), after the run and its reanalysis, with the frozen functions of code/mag_s3c.py; nothing here changes a verdict. The script is explore_after_run.py, and its outputs are the JSON and log files of this folder.

## 1. The scale of CISS of Kobori et al. (2026)

**Correction (v0.3.1).** Kobori et al. define the CISS efficiency as 100(1 − cos χ) for the state cos(χ/2)|S⟩ + sin(χ/2)|T₀⟩, so neither reading of this section, sin χ = 0.06 or sin²χ = 0.06, matches their 6% (χ = 20°). The values below stay correct as functions of the angle; the comparison at their definition is in section 3 and in note 1 of the notes after release in ../../README.md.

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

The response of form K grows almost exactly as sin χ cos χ, the coherence term of its exact split, so the amplitude reading moves W_C to the threshold of the dipole test at p = 1 and S = 1 even with full non-reciprocity, while W_D stays 2–4 times above it; axial disorder multiplies the response by g(S), 1.66 at S = 0.4, when the neurons keep their modulation depth (note 5 of the notes after release in ../../README.md). The triplet-share reading keeps the margins close to those at 0.1π. P-0 at sin χ = 0.06 gives 0.030–0.042 in W_C and 0.036–0.052 in W_D for the axis z without relaxation and at 10⁶ s⁻¹, and 0.005 at dephasing 3·10⁷ s⁻¹ (scale.json).

## 2. H3 in the trial run of 10 November

The sealed line of the trial run (substitute rates, 60 directions) had H3 not met in W_C and W_D, and the run has it met. On 60 directions the harmonic fit of degree 10 is underdetermined: 66 even and 55 odd coefficients against 30 antipodal pairs, and least squares returns the solution of smallest norm. The script solves the reference anew with the frozen solver on the grid of 60 directions and rebuilds form K at χ_b = 0.1π, m = 1, axis r̂, on the levels of H3.

| Grid and rates | W_C: F | W_C: F_amp | W_D: F | W_D: F_amp |
|---|---|---|---|---|
| 2000 directions, Cry4a rates (the run) | 0.47–0.71 | 0.050–0.060 | 3.9–4.9 | 0.153–0.184 |
| 60 directions, Cry4a rates | 0.0001 | 0.037–0.045 | 0.0002–0.0003 | 0.115–0.139 |
| 60 directions, substitute rates (the trial) | 0.0001 | 0.053–0.062 | 0.0003–0.0004 | 0.155–0.186 |
| 2000 directions, substitute rates | 0.61–0.87 | 0.072–0.084 | 3.2–3.8 | 0.207–0.248 |

On 60 directions the even part spreads over the degrees the grid cannot resolve, the even range of the receptor grows, and the ratio transfer F falls to about 10⁻⁴, below the threshold; the amplitude transfer, which takes only the dipole, keeps 74–76% of its value on 2000 directions. On 2000 directions the substitute rates raise F_amp by about 40%, and H3 holds there with a margin of 7–8 times in W_C and 21–25 times in W_D. The grid of 60 directions thus explains why H3 failed in the trial: with the frozen code, the same grid and rates give H3 not met. The trial itself ran on code from before the freeze, so this recomputation is no replay of it.

## 3. Plan of the computations in Colab after release v0.3.0

Written on 12 November 2026 (UTC) and committed before anything of it was computed. The script is
kobori_and_dephasing.py and the notebook notebooks/MAG-S3c_colab_after_release.ipynb; they use the frozen code of
code/ unchanged and check its SHA-256 against FREEZE.txt first. The results go into the notes after release v0.3.1 and
change no verdict.

**Why.** A comment on release v0.3.0 pointed out that Kobori et al. define the CISS efficiency as 100 (1 − cos χ). Their
preprint (ChemRxiv, doi 10.26434/chemrxiv.15000186/v2) fits the time-resolved EPR spectra of the cryptochrome-DASH of
Xenopus laevis with the singlet-born state cos(χ/2)|S⟩ + sin(χ/2)|T₀⟩ (their Eq. 6), recombination at χ_CR = −χ
(Eq. 16) and the CISS axis along the axis of the dipolar coupling, with χ = 20° (6%) for the pair FAD–W_C and 23° (8%)
for FAD–W_B. Neither reading of section 1 matches this definition. Among the forms of MAG-S3c their model is P-LH with
the CISS axis r, and the run computed the polarisation forms at χ = 0.1π = 18°, an efficiency of 4.9% in this measure.

**A. Polarisation forms at the angles of Kobori et al.** Direct cells of P-LH and P-same at χ = 20° and 23° in W_C and
W_D, on the axes r, z and x, without relaxation, at 10⁶ s⁻¹ of random fields and of dephasing, and at 3·10⁷ and
10⁸ s⁻¹ of dephasing: 120 cells on the grid of the run (2000 directions, Cry4a rates). Form K at equal triplet
population (χ_b = χ/2) and the control P-0 at χ follow from the references of the run without new solves.

**B. Dense dephasing.** References of W_C and W_D at dephasing 4·10⁷, 5·10⁷, 6·10⁷ and 8·10⁷ s⁻¹, between the levels
3·10⁷ and 10⁸ s⁻¹ of the run: 8 references. With them k½ of form K (χ_b = 0.05π and 0.1π, m = 1) on the axes z, x and
r follows by the rule of the protocol, the first crossing of half, linear in lg k between adjacent levels, and |d_odd·z|
at 6·10⁷ s⁻¹ needs no interpolation.

**C. Exchange coupling (added on 12 November 2026 before it was computed).** Kobori et al. fitted J = 22 μT for the
pair FAD–W_C (2J = E_S − E_T₀), where the run took J = 0. The script kobori_exchange.py adds the exchange term in the
convention of RadicalPy, −J|γ_e|(2 S_A·S_B + 1/2), to the frozen systems W_C and W_D with J = +22 μT and −22 μT and
computes P-LH and P-same at χ = 20° on the axis r, without relaxation and at 10⁶ s⁻¹ of both kinds: 24 cells. The
frozen code stays unchanged, and a check cell at J = 0 must equal the cell of the frozen W_C bit for bit. Because
Colab did not connect, the core of A on the axis r (32 cells without the level 10⁸ s⁻¹) and part C run on the
calibration machine; the rest of A and part B wait for Colab.

**Checks.** Twelve direct cells (P-LH and P-same at χ = 0.1π on the axis r, without relaxation and at 10⁶ s⁻¹ of both
kinds, in W_C and W_D) repeat cells of the run, and the script stops before A and B when one of them differs from the
archive of the run by more than 10⁻⁹ relative. The reference of W_C at dephasing 3·10⁷ s⁻¹ is compared at the end. On
the calibration machine the first of the twelve came out bit for bit, and the rule for k½ reproduces k_half.csv to
4·10⁻¹³ relative.

**Reading.** For every cell the summary gives F, F_amp and the stricter of the two at p = 1 and S = 1, as in the run,
and calls a cell visible to the dipole test when the stricter response is at least 0.01. The comparison with Kobori et
al. reads P-LH on the axis r at 20° in W_C first, then W_D, the other axes, P-same, K and P-0. Section B reports k½
beside the value from the levels of the run and the class of the protocol (below 3·10⁷, 3–6·10⁷, above 6·10⁷ s⁻¹).

**Cost.** About 43 processor hours, 1–1.5 hours on the 44 processors of a TPU v6e-1 runtime in Colab.

**Results so far (12 November 2026, UTC).** The core of A on the axis r (32 cells: P-LH and P-same at 20° and 23° in W_C and W_D, without relaxation, at 10⁶ s⁻¹ of both kinds and at dephasing 3·10⁷ s⁻¹) and part C (24 cells) ran on the calibration machine; after_release/ holds the driver local_core.py, core_summary.py, the logs, the summaries and the raw cells in local_cells.tar.gz. The four check cells at χ = 0.1π without relaxation equal the run bit for bit, and so does the check cell of part C at J = 0. The numbers are in notes 1 and 2 of the notes after release in ../../README.md. The rest of A (the axes z and x, dephasing 10⁸ s⁻¹), the remaining check items and part B wait for Colab.

**Part B (13 November 2026, UTC).** after_release/local_dense.py, written before the run (SHA-256 b65285266e85442e0f7b4a07eee6a5366b9cd1fa40a513199ba57e151d3bec04) and committed as 4ad2c8f while the run was in progress, before any of its results was read, computed the 8 references of part B and the check reference of W_C at dephasing 3·10⁷ s⁻¹ on the calibration machine with 2 processes in 1.6 hours. The check reference equals the archive of the run bit for bit, and the rule for k½ on the levels of the run reproduces k_half.csv to 4·10⁻¹³ relative. after_release/ holds local_dense.log, local_dense_summarize.log, local_dense_summary.json and the raw references in local_dense_refs.tar.gz, whose pair_done.json covers the second call of the pair step, the references of W_D. With the dense levels k½ of form K lies 1–7% below its value on the levels of the run in all 10 combinations of W_C and W_D where it lies below 10⁸ s⁻¹; W_C on the axis r moves from the class above 6·10⁷ s⁻¹ into the class 3–6·10⁷ s⁻¹, and W_C on the axis x stays in that class. The table and the reading are in note 10 of the notes after release in ../../README.md. The rest of A and its remaining check items still wait for Colab.


## 4. The recombination rate of W_C (after release v0.3.2)

The Cry4a rates of the main grid take the recombination rate that Xu et al. (2021) estimated for FAD–W_D, 3.4·10⁵ s⁻¹, for both pairs; Wong et al. (2021) cite about 1.2·10⁷ s⁻¹ for FAD–W_C and a forward step of 5–10·10⁶ s⁻¹ for both. after_release/wc_rate.py was written at 16:53:58 UTC on 12 November 2026 (SHA-256 261317aa3e5619c729f2226c9e593328d228c73e9218c0ccd8b726fe645adcbc) before the run and committed as 3a1ca48 while the run was in progress, before any of its results was read. It repeats the cells of note 1 in W_C on the axis r at χ = 20° with k_R = 1.2·10⁷ s⁻¹ and k_F = 5·10⁶ s⁻¹: P-LH and P-same directly, K and P-0 from references at the same rates, without relaxation and at 10⁶ s⁻¹ of random fields and of dephasing. The new rate set is added to S3C_RATES at run time, and the frozen files are unchanged. Two cells of the run (P-LH and P-same at χ = 0.1π on the axis r without relaxation) were recomputed first and equal the archive bit for bit. The run took 16 minutes on the calibration machine with 2 processes; after_release/ holds wc_rate.log, wc_rate_summary.json and the raw cells and references in wc_rate_cells.tar.gz. The numbers are in note 9 of the notes after release in ../../README.md.

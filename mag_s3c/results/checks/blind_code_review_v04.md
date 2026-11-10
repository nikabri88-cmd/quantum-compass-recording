# MAG-S3c blind code review of version 0.4, 10 November 2026

An independent reviewer (a separate agent that had not seen the code being written) received the changes from version
0.3 (commit d5e96ec) to version 0.4 (commit 8c89ca8) and the text of protocol v0.4, with the rule that it computes
nothing the hypotheses, expectations or maps use. It computed no pattern of a full pair with CISS or with relaxation
at any rate and nothing at the Cry4a rates, at the rates of E12 or at the hop rates of E13. It used geometry only,
the small pair at the MAG-S2 rates, the small composite and the full composite without CISS at the substitute rates
(iterations, residuals and the exact zero of I10). It modified no file. Every finding was fixed before calibration
round 3 (see the fix column).

| # | Severity | Finding | Fix |
|---|---|---|---|
| B1 | blocker | The crystal 222 of map E14 and of I11 was averaged about the transposed frame of the flavin, whose axes lie 64–89° away from the cell axes of 6PU0; I11 and the crystal test used the same axes and could not see it | the cell axes are the columns of the frame matrix (R e_a, R e_b, R e_c); C4 and a new test build the four symmetry mates of chain A from the operators of P2₁2₁2₁ and require that the field rotation between molecule 1 and each mate equals an element of the crystal average to 10⁻¹² (found 2.9·10⁻¹⁵) |
| B2 | blocker | Cell 3 of the notebook could not pass: the fast tests call scripts/make_colab_notebook.py, which the notebook did not embed (the same gap stood in v0.3) | the builder is embedded with the other files, so its SHA-256 belongs to the build identity |
| I1 | important | The test of the composite pair compared the two-site solver with a dense solve of the same model objects | a new test builds site W_D directly in one common ordering, takes the free N1 from RadicalPy's own nuclear Zeeman term, treats the hops as the identity map and solves densely; it agrees with composite_solve to 10⁻¹² for both forms, both recombination schemes and both preconditioners |
| I2 | important | mag_s3c.py ran at the Cry4a rates on a bare flag, and pipeline.sh accepted a confirmation listing any one correct file and ignored the build identity and the blind marker | --confirmatory takes the confirmation.json of cell 5; mag_s3c.py and pipeline.sh require the answer «запускай», every file of the study with its SHA-256 as on disk, the build identity of the confirmed hashes and the marker of the passed blind check of the same build, which cell 5 now copies into the confirmation; tests cover a subset of files, a wrong identity, a missing marker and a wrong answer |
| M1 | minor | Parts of the composite cells carried no provenance | every part records the SHA-256 of mag_s3c.py, the rates, the hop rates and the grid, and the merge stops when a part disagrees |
| M2 | minor | The validity of a run did not include the counts of the comparisons | a run is valid only with 184 comparisons and 8 known in advance |
| M3 | minor | The post-check of the smoke and blind runs did not require the pair marker, the substitute rates or the intended grids | results.json reports the grids and every rate pair and hop scale seen; the post-check requires the pair marker, the substitute rates throughout and the grids of the check |
| M4 | minor | --grid-half and --t-max were accepted with --confirmatory | refused |
| M5 | minor | The order of the composite jobs did not follow their cost | the jobs go by the cost measured in round 3, the hops 10⁸/10⁸ first |
| M6 | minor | The analysis kept only quantiles of k½ over the CISS axes and the rate of E10 only for the axis r̂ | maps_axes.npz beside results.json holds k½ with its class for every axis of E5 and the rate of E10 for every axis in W_C and W_D |
| M7 | minor | first_below gave a misleading note when s_C is undefined without relaxation | a note of its own |
| M8 | minor | Composite rows carried an underdetermined check fit of degree 24 on 500 directions | no check fit for the composite pair (no rule used it) |
| M9 | minor | Some maxima could drop a NaN | NaN-safe maxima in the invariants and the checks |
| M10 | minor | env_check allowed ±2 for every count of GMRES iterations | ±2 or 2% of the count |
| M11 | minor | Behaviour without tests | tests of the gate, of GMRES and the composite solver failing to converge, of env_check and of the flag of the zero circle |

The reviewer found no defect in the physics or the numerics apart from B1: the reconstruction identity and its
signs, the split of K, form E and C6, the relaxation superoperators and the solver, the composite model (rebuilt
independently, agreement to 2·10⁻¹⁴), both preconditioners with the switch at 4·10⁸ s⁻¹, the real Hermitian GMRES,
the exact zero of I10 (margin above 400), the geometry and the dipolar tensor, the maps (E5 and the circle of H2
against the frozen summarize() of rebuilt cells to 4·10⁻¹² relative, E14 against numerical azimuthal averages to
3·10⁻¹⁴). Its coverage list also includes the design counts (574, 442, 54, 24, 146, 24, 24 and 12), the substitute
rates, every threshold and sense of I1–I11, Q1–Q4 and H1–H5, the comparisons known in advance, k½ with its classes,
the conditions of a valid run, the gates of the blind and confirmatory runs, the deterministic notebook build and
the frozen files, identical to the release MAG-S3b v0.2.0.

# MAG-S3c blind code review, 9 November 2026

An independent reviewer (a separate agent that had not seen the code being written) received a snapshot of the code
folder and the protocol draft v0.3, with the rule that it computes nothing the hypotheses, expectations or maps use.
It used about 80 s of CPU, computed nothing at the Cry4a rates on W_C or W_D with CISS or relaxation, and did not run
the pipeline. Its report follows unchanged in substance; every finding was fixed before the second calibration round
(see the status column).

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | blocker | The blind check returned exit 0 whatever its summary showed, and the markers CELL3_OK and BLIND_OK were never cleared or tied to the notebook build, so cell 5 could accept a failed or old blind check | the summary of smoke and blind runs exits 1 unless every invariant is ok and the result is complete and well formed; cells 3 and 4 write markers that hold the identity of the embedded file set, cell 4 deletes BLIND_OK before the check, and cell 5 re-hashes the files and accepts only a marker of the same build |
| 2 | should fix | Nothing in the code stopped a run at the Cry4a rates before the freeze | pair, checks and curves refuse to run without --blind-rates or --confirmatory; pipeline.sh passes --confirmatory only when confirmation.json, written by cell 5 after the answer «запускай», holds that answer and the SHA-256 of the files in the code folder |
| 3 | should fix | The notebook build was not reproducible (gzip time stamp, member times and owners) | sorted members with zero times and owners, gzip with mtime 0; a test builds twice and compares the hashes |
| 4 | should fix | The metadata hashed only the three own files and recorded protocol version 0.2 | every step records the SHA-256 computed at run time of all files of the study and of its harness; version 0.3; validity requires one BLAS thread in every step |
| 5 | should fix (minor) | results.json, the done marker of the analysis, was written before the analysis finished | results.json is written last; an incomplete run gets results_incomplete.json and exit 1 |
| 6 | minor | C3 judged convergence vacuously and lost the diagnostics of the C1 cells on a resumed calibration | the diagnostics of each C1 grid are stored with its yields and merged on resume; the number of GMRES restarts is reported |
| 7 | minor | I4 and I5 left out the residuals and conservation errors of the 31 small-pair solves | merged into the diagnostics of the checks |
| 8 | minor | C4 targets differed from the rounded angles in the protocol text | the protocol gives every angle with one decimal (89.3°, 57.3°, 77.6°, 46.3°), and C4 also checks the S2 axes and the listing angle without the change of sign |
| 9 | minor | Some tests were weaker than their claims | RadicalPy superoperators to 1e-15 k; all three nuclei and axes of the E7 systems; a cell job under the substitute rates records them |
| 10 | minor | Validity compared counts instead of key sets | key sets of I2, I3, I5 and I6 and the full set of curves (cases, three kinds, length, step, output interval) |
| 11 | minor | The H4 halving map gave a meaningless bracket where the denominator vanishes | those combinations are marked as excluded |
| 12 | minor | A NaN could pass the diagnostics silently | solve() raises on a non-finite solution and the merges keep a NaN |

The reviewer found no defect in the physics, the solver or the scoring of H1–H5, Q1–Q4 and I1–I7. Its coverage list:
frozen files and I1; the frame, tensors and point dipole (d = 0.340 and 0.193 mT); the solver equation, Schur/trsyl
preconditioner and GMRES; the dense Liouvillian, exact-step propagator and RK4 generator; the forms K, K-inc, P-same
and P-LH; 534 unique cells, 18 E4 cells, 12 I6 cells, 18 curves, 135 check sets and 31 small-pair cases; the
substitute rates reaching every solve; the receptor transfer, identical to MAG-S3b; every threshold and comparison
direction of I2–I7, Q1–Q4, H1–H5, the denominator rule and the maps; the tests (none computes a forbidden number);
the pause handling; the output of the smoke and blind runs; BLAS pinning in every entry point.

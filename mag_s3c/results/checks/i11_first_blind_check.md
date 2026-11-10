# MAG-S3c: invariant I11 in the first blind check in Colab, 10 November 2026

The blind pipeline check of cell 4 (notebook build 895803767c9c0b57, TPU v6e-1, 44 processes) ran every step with exit
code 0 from 16:05:49 to 16:52:50 UTC. Ten invariants held, the cells and the composite pair were complete, and the
grids and the substitute rates were as intended, but I11 returned "error". The check therefore ended with CHECK FAILED
(exit code 1), deleted its output unread and wrote no marker, so cell 5 stayed closed. A first attempt from 15:22:43 UTC
had stopped after the pair step when the Colab tab lost its connection and the runtime was replaced. The logs of cells
3 and 4 are colab_build_8958037_*.

## What I11 checks

The average F over the four rotations of the crystal 222 keeps only the odd terms of the form u v w g(u², v², w²) in
the coordinates of the cell, so its odd part vanishes for fields in the three cell planes and changes sign at each cell
mirror. Version 0.4 divided the residuals of these two identities by the largest odd value of the crystal average
itself (floored at 1e-9 of its even range) and required less than 1e-12.

## Diagnosis

Everything below ran at the substitute rates of the blind check and printed only pass or fail, residuals, iterations
and times.

- The averaging is exact to rounding. In 228 pair cells recomputed on the 500 directions of the blind check (every
  direct cell of W_C and W_D without relaxation, the rebuilt cells of W_C and W_D without relaxation and at dephasing
  1e8 s^-1, of E12 and of the variants of E3 and E7) and in 12 composite cells on its 100 directions, the residuals stay
  at or below 1.8e-15 and 2.5e-15 of the largest odd value of the molecule.
- The statistic as written divides rounding by what the crystal keeps of the odd part. The 222 average removes the
  dipole, which carries most of the odd part, so it keeps a small share, and the statistic grows as the inverse of that
  share. On the 60 directions of the smoke runs the fit of degree 10 is underdetermined and spreads the odd part over
  higher degrees: for the rebuilt cells of W_C without relaxation the statistic was 2.6e-15. On the 500 directions of the
  blind check the same cells give 1.8e-13, and the same cells of W_D, of one rate set of E12 and of two variants give
  1.1e-13 to 2.7e-13, within a factor of 4 to 9 of the threshold, while the residuals over the molecule stay at 1.2e-15
  to 1.5e-15.
- None of the recomputed cells fails I11 as written. The failing cells of the blind check lie among the cells that two
  processors cannot recompute in a few hours (direct cells with relaxation, most rebuilt cells with relaxation,
  composite cells with the fastest hops); the crystal of the blind check held 844 cells. The second blind check prints
  both statistics.
- A synthetic pattern shows the mechanism: when the crystal keeps a share e of its natural odd part, the statistic as
  written is 1.1e-13 at e = 1, 1.1e-11 at 1e-2 and 1.1e-8 at 1e-5, while the residual over the molecule stays at 5e-16.

## Change before the freeze

- I11 divides the residuals by the larger of the largest odd values of the molecule and of the crystal, floored at
  1e-9 of the even range of the crystal as before; the threshold stays 1e-12 (analyze_s3c.i11_ratios). The record of
  I11 keeps the statistic of the first check as worst_over_crystal_odd_max without a verdict, and the summary of the
  smoke and blind checks prints both.
- A new test with known answers, test_i11_normalization: a pattern whose crystal keeps 1e-6 of its natural odd part
  passes with residuals below 1e-13 of the molecule while the statistic as written exceeds 1e-12; an average about
  axes turned by 1e-3 rad from the planes of the test gives 5e-7, an average over three of the four rotations gives
  0.25, and a cell without its molecule gives NaN, and all three fail.
- Cells 3, 4 and 5 of the notebook now end with an error when their exit code is not 0 (or 3 for a pause of cell 4), so
  that Colab no longer shows a completed cell after a failed check.
- The protocol states the new denominator in the row of I11 and in the list of clarifications of version 0.4.
  Hypotheses H1-H5, their thresholds and the rules of the analysis did not change. The calibration does not depend on
  the changed files (analyze_s3c.py, test_s3c.py, scripts/pipeline.sh, scripts/make_colab_notebook.py), so the reference
  calibration of round 3 stays; the new notebook is build f994d9b191aa9dbb.

## Second blind check

On build f994d9b191aa9dbb cell 3 ran the 30 fast tests and gave the same comparison of the environment (192 identical,
31 within the tolerances, none different), and the blind check ran from 18:43:46 to 19:31:51 UTC with every step at exit
code 0. All eleven invariants held. Over all 844 cells of the crystal, I11 gave 2.5e-15 for the plane and the mirror
residuals over the odd part of the molecule, while the normalization of the first check gave 5.2e-12 and would have
failed again. The check passed, deleted its output unread and wrote its marker for the build (colab_build_f994d9b_*).

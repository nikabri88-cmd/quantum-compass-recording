# MAG-S3c: real pair geometry and the decay of coherence

A pre-registered simulation study of whether the two main conclusions of MAG-S3b survive the real geometry of the radical pair in cryptochrome 4 and spin relaxation, and how much of the odd part of the compass code, the part that changes sign when the field is reversed, the singlet–triplet coherence itself carries.

Author: Julia Ryzhkova (Siriona, siriona.one; ORCID 0009-0005-4194-6059).
Status: frozen before the confirmatory run on 10 November 2026 at 19:40 UTC. The protocol (v0.4, in Russian), the code, the requirements, the structure file, the reference calibration and the Colab notebook are listed with their SHA-256 in FREEZE.txt, and the hashes were posted in Telegram (https://t.me/attractordm) before the run. The results will follow in release v0.3.0 of this repository. MAG-S3c builds on MAG-S3b (this repository, release v0.2.0, https://doi.org/10.5281/zenodo.24079875), MAG-S3 (release v0.1.0, https://doi.org/10.5281/zenodo.24043500) and MAG-S2 (release v0.1.2, https://doi.org/10.5281/zenodo.24030614) and imports their code unchanged.

## The question

MAG-S3b found, on the three-nucleus MAG-S2 pair without relaxation and at J = D = 0, that the coherence form K of chirality-induced spin selectivity (CISS) gives 253 times more odd part than the polarisation forms, and that the projection of the odd dipole on the receptor axis survives a CISS axis across the receptor, because the hyperfine tensors set its direction. MAG-S3c recomputes the same pair in three geometries: S2 (the tensors of MAG-S2), and the FAD–W_C (W318) and FAD–W_D (W369) pairs of pigeon cryptochrome 4 (ClCry4a) from PDB 6PU0, with the dipolar coupling of each pair. Two kinds of spin relaxation, singlet–triplet dephasing and random fields, run up to 10⁸ s⁻¹, the dephasing measured in Cry4a by Golesworthy et al. (2023).

Five hypotheses get a verdict per geometry, from 184 comparisons, 8 of which follow from published cells of MAG-S3b and are reported without a verdict:

1. **H1.** The coherence form stays at least 10 times stronger than the larger polarisation form, in ε_mol and in |F|, in each of 40 combinations of geometry, CISS axis and relaxation up to 10⁶ s⁻¹.
2. **H2.** In W_C and W_D the median |d_odd·z| over a circle of 360 CISS axes across the receptor is at least half its value at the axis z.
3. **H3.** At Δχ = 0.1π with the CISS axis along the radical-pair axis, the responses F and F_amp of the real pairs reach the threshold 0.01 of the dipole test.
4. **H4.** Relaxation at 10⁶ s⁻¹ keeps at least half of d_odd·z, for both kinds, in every geometry and on every axis.
5. **H5.** The exact split of form K gives the coherence a share s_C of at least 0.9 of d_odd·z.

The main descriptive quantity is k½, the relaxation rate at which the odd dipole falls by half, compared with the measured dephasing of 3–6·10⁷ s⁻¹. Exploratory maps E1–E14 add the microscopic CISS form of Denton et al. (2026), fast recombination of the pair, the composite pair W_C ⇌ W_D with hops of the hole, the sphere of CISS axes and the average over the crystal 222 of 6PU0.

## Contents

| Path | What it holds |
|---|---|
| FREEZE.txt | the freeze record: time, checks in Colab and the SHA-256 of the 16 frozen files |
| protocol/MAG-S3c_protocol_v0.4_ru.md | the frozen protocol |
| code/ | mag_s3c.py (pair in three geometries, relaxation, references and reconstruction, composite pair, receptor and maps, calibration), analyze_s3c.py (invariants I1–I11, expectations Q1–Q4, hypotheses H1–H5, k½, maps E1–E14), test_s3c.py (31 tests with known answers), scripts/ (pipeline, environment check, notebook builder), requirements.txt, data/6PU0.pdb and the frozen MAG-S3b v0.2.0 files it imports |
| notebooks/MAG-S3c_colab.ipynb | the notebook of the run, with the code and the reference calibration embedded with their SHA-256 |
| results/calibration/ | calibration C1–C6 in three rounds; round3 is the reference on the final code, with its tests and smoke run in round3/checks |
| results/checks/ | the blind code reviews of versions 0.3 and 0.4; the tests, environment checks and blind pipeline checks in Colab of builds 8958037 and f994d9b; i11_first_blind_check.md on the change of I11 after the first blind check |

## Reproduce

Python 3.13.18 with `pip install -r code/requirements.txt` (NumPy 2.5.3, SciPy 1.18.1, RadicalPy 1.0.9). From code/:

```
python3 -I test_s3c.py                                       # 31 tests with known answers (--fast skips one)
python3 -I mag_s3c.py calibrate --out CAL --workers N        # calibration C1–C6
python3 -I scripts/env_check.py ../results/calibration/round3 CAL
bash scripts/pipeline.sh blind OUT                           # blind check at substitute rates, output deleted
```

The confirmatory run (`bash scripts/pipeline.sh confirmatory OUT`) needs OUT/confirmation.json, which cell 5 of the notebook writes after the answer «запускай»; without it the code computes nothing at the rates of the hypotheses.

## Licence

Code is released under the MIT licence (LICENSE at the root of the repository); text, the protocol and the results are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

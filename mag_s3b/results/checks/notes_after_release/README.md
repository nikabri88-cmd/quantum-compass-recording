# Checks of the notes after release (v0.2.1)

These scripts check the exact statements of "Notes after release (v0.2.1)" in mag_s3b/README.md. They use invented toy radical pairs: two electrons, two spin-1 nuclei (toy_pair.py) or three spin-1/2 nuclei (two_site_parity.py), random anisotropic tensors, a point dipole, 50 µT and toy rates. Nothing here uses the W_C or W_D geometry, the RadicalPy tensors or the Cry4a rates, and the logs print residuals and pass/fail only.

| Script | Note | What it checks |
|---|---|---|
| two_site_parity.py | 1, 2, 3 | a two-site chain (bridge and terminal pair) written independently of the review: parity when spin-selective recombination happens only at the birth site through the birth projector (with CISS at birth and at the hop, dephasing, no back hop); the odd part without CISS when both sites recombine, against the bridge lifetime; the operator structure of Denton et al. (2026); a second solver without a Liouville matrix |
| tr_condition.py | 1 | the condition on the birth projector: form K keeps parity when born and recombining through the same projector, a polarisation projector does not |
| reciprocal_coherence_zero.py | 3 | a single pair born and recombining through cos θ\|S⟩ + i sin θ\|T₀⟩ keeps parity; mismatched birth and recombination break it |
| crystal_222.py | 4 | the mean over the four orientations of a P2₁2₁2₁ cell: no odd part for fields in the cell planes, a sign change under each cell mirror, a nonzero odd part off the planes |
| reconstruction_check.py | 5 | the reconstruction Φ_R(−n; ρ₀) = k_R Tr[Θρ₀Θ⁻¹ X_S(n)] for eleven birth states, the zeros of the singlet-born pair and of K-iso, and the split of form K into coherence and alignment terms |

Run each script from this folder with `python3 <script>` (NumPy and SciPy). The logs were written with NumPy 2.5.3 and SciPy 1.18.1; two_site_parity.py takes about ten minutes on two cores, the others under two minutes.

The subfolder independent_review/ holds the original toy checks of the independent review (Claude Fable 5.2), with its own README in Russian: the two-site model with its four runs and a step-by-step Runge–Kutta check of the odd part without CISS. The script two_site_parity.py here reproduces their results on separate code.

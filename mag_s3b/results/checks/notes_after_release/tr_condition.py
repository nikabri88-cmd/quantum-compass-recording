"""Note 1 of v0.2.1, the condition on the birth projector: parity needs a birth projector that is invariant under
time reversal. Born and recombining through the same projector, form K keeps parity, and a polarisation projector
(real singlet-triplet coherence) does not. Toy pair of toy_pair.py; residuals and pass/fail only."""
import numpy as np
from toy_pair import form_P, form_K, yield_F, time_reversed, random_directions, unit

n_c = unit([0.3, -0.5, 0.8])
QP, QK = form_P(0.1 * np.pi, n_c), form_K(0.1 * np.pi, n_c)
dirs = random_directions(4, seed=3)


def odd_rel(rho0, Q):
    o = max(abs(yield_F(n, rho0, Q) - yield_F(-n, rho0, Q)) for n in dirs) / 2
    return o / max(abs(yield_F(n, rho0, Q)) for n in dirs)


print("birth state of form K invariant under time reversal:", np.allclose(time_reversed(QK), QK))
print("polarisation birth state invariant under time reversal:", np.allclose(time_reversed(QP), QP))
print(f"born and recombining through the same K projector: worst |odd| / yield = {odd_rel(QK, QK):.1e}")
print(f"born and recombining through the same polarisation projector: odd part nonzero = {odd_rel(QP, QP) > 1e-9}")

"""Note 5 of v0.2.1: reconstruction of every singlet-recombining form from the singlet-born solution, and the exact
split of the odd part of form K into a coherence term and an alignment term (found by an independent review,
Claude Fable 5.2, and checked here on a toy pair). Residuals only.

Reconstruction (time reversal Theta, Phi_R the recombination yield, X_S(n) the time integral of rho for a pair born in
the singlet at field direction n):   Phi_R(-n; rho0) = kR Tr[Theta rho0 Theta^-1 X_S(n)]   for every birth state
rho0 of a pair recombining through P_S, so one solution per field direction gives every such form and CISS axis.

Split of form K at m = 1, rho_K = cos^2 chi P_S + sin^2 chi s P_S s + i sin chi cos chi (P_S s - s P_S),
s = n.sigma_A: the singlet term has no odd part, so
odd(K) = sin chi cos chi odd(C) + sin^2 chi odd(A),  C = i(P_S s - s P_S) (linear in the CISS axis),  A = s P_S s,
and the incoherent control K-inc = P_S rho_K P_S + P_T rho_K P_T keeps only sin^2 chi odd(A); the isotropic control
K-iso = cos^2 chi P_S + sin^2 chi P_T/3 has no odd part.
"""
import numpy as np
from toy_pair import (PS, PT, ONE, KR, form_K, form_P, sigma_A, solve_integral, yield_R, time_reversed,
                      random_directions, unit)

chi = 0.1 * np.pi
axes = [unit([0.2, -0.4, 0.9]), unit([0.8, 0.5, -0.1]), unit([-0.3, 0.9, 0.3])]
dirs = random_directions(6)

forms = {"singlet": PS, "K-iso": np.cos(chi) ** 2 * PS + np.sin(chi) ** 2 * PT / 3}
for j, n_c in enumerate(axes):
    rho_K = form_K(chi, n_c)
    forms[f"K, axis {j + 1}"] = rho_K
    forms[f"K-inc, axis {j + 1}"] = PS @ rho_K @ PS + PT @ rho_K @ PT
    forms[f"P, axis {j + 1}"] = form_P(chi, n_c)

# 1. reconstruction against direct solves at -n
worst_rec, scale = 0.0, 0.0
for n in dirs:
    XS = solve_integral(n, PS)
    for rho0 in forms.values():
        direct = yield_R(-n, rho0)
        rebuilt = KR * np.trace(time_reversed(rho0) @ XS).real
        worst_rec = max(worst_rec, abs(direct - rebuilt))
        scale = max(scale, abs(direct))
print(f"1. reconstruction from the singlet-born solution, {len(forms)} birth states x {len(dirs)} directions: "
      f"worst |direct - rebuilt| / yield = {worst_rec / scale:.1e}")


def odd(n, rho0):
    return (yield_R(n, rho0) - yield_R(-n, rho0)) / 2


# 2. exact zeros
z = max(abs(odd(n, forms[k])) for n in dirs for k in ("singlet", "K-iso"))
print(f"2. odd part of the singlet-born pair and of K-iso: worst |odd| / yield = {z / scale:.1e}")

# 3. split of K into coherence and alignment, and K-inc
worst_split = 0.0
for n_c in axes:
    s = sigma_A(n_c)
    C, A = 1j * (PS @ s - s @ PS), s @ PS @ s
    rho_K = form_K(chi, n_c)
    rho_inc = PS @ rho_K @ PS + PT @ rho_K @ PT
    for n in dirs:
        oC, oA = odd(n, C), odd(n, A)
        worst_split = max(worst_split,
                          abs(odd(n, rho_K) - np.sin(chi) * np.cos(chi) * oC - np.sin(chi) ** 2 * oA),
                          abs(odd(n, rho_inc) - np.sin(chi) ** 2 * oA))
print(f"3. odd(K) = sin chi cos chi odd(C) + sin^2 chi odd(A) and odd(K-inc) = sin^2 chi odd(A): "
      f"worst residual / yield = {worst_split / scale:.1e}")

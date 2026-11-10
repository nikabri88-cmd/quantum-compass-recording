import numpy as np, sys
from two_site import *
chi1, n1 = 0.1 * np.pi, sph(72, 25)
chi2, n2 = 0.07 * np.pi, sph(40, 300)
Pb = projK(chi1, n1); U2 = UK(chi2, n2)
base = dict(k2=3e7, km2=4e6, kCR=1.5e7, kSP=1e6)
print("rates (toy):", base, " kCR2 where used: 2e6")
odd_test("T1 no CISS, recombination only from the bridge (P_S)", Pb=PS, PCR=PS, PCR2=PS, U2=ONE, kCR2=0.0, **base)
odd_test("T2 no CISS, singlet recombination from both sites", Pb=PS, PCR=PS, PCR2=PS, U2=ONE, kCR2=2e6, **base)
odd_test("T2b same, bridge Hamiltonian = Zeeman + exchange only", Pb=PS, PCR=PS, PCR2=PS, U2=ONE, kCR2=2e6, aniso_bridge=False, **base)
odd_test("T3 CISS (U1 at birth, U2 at hop), P_CR = P_b, kCR2 = 0", Pb=Pb, PCR=Pb, PCR2=PS, U2=U2, kCR2=0.0, **base)
odd_test("T3r same with S-T dephasing 2e7 at both sites", Pb=Pb, PCR=Pb, PCR2=PS, U2=U2, kCR2=0.0, kstd=2e7, **base)
odd_test("T4 as T3 but P_CR = P_S (angle mismatch at the bridge)", Pb=Pb, PCR=PS, PCR2=PS, U2=U2, kCR2=0.0, **base)
odd_test("T5 as T3 plus recombination from CS through P_S", Pb=Pb, PCR=Pb, PCR2=PS, U2=U2, kCR2=2e6, **base)
odd_test("T6 as T3 plus recombination from CS through U2 P_b U2^+", Pb=Pb, PCR=Pb, PCR2=U2 @ Pb @ U2.conj().T, U2=U2, kCR2=2e6, **base)
b0 = dict(base); b0["km2"] = 0.0
odd_test("T7 as T3 with no back hop (km2 = 0)", Pb=Pb, PCR=Pb, PCR2=PS, U2=U2, kCR2=0.0, **b0)

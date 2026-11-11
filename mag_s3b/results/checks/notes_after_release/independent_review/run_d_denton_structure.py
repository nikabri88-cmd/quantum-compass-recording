"""Denton-type structure in the toy: the same CISS projector P_theta at birth and in both recombination channels,
spin-conserving hops, bridge Hamiltonian = de*P_theta + 2J*P_S (no Zeeman, no hyperfine), terminal site with
Zeeman + anisotropic hyperfine + dipolar. Invented numbers; rates are toy values."""
import numpy as np
import two_site as ts
from two_site import *
theta, n = 0.1 * np.pi, sph(72, 25)
Pth = projK(theta, n)
orig = ts.hams
def mk(J, de):
    def h(bv, **kw):
        Hb, Hcs = orig(bv, **kw)
        return GAM * (de * Pth + 2 * J * PS), Hcs
    return h
base = dict(k2=3e8, km2=4e6, kCR=1.5e7, kSP=1e6, Pb=Pth, PCR=Pth, PCR2=Pth, U2=ONE)
for label, J, de, kCR2 in [("D1 theta!=0, J=0,  de=0,  kCR2=2e6", 0.0, 0.0, 2e6),
                           ("D2 theta!=0, J=0,  de=5,  kCR2=2e6", 0.0, 5.0, 2e6),
                           ("D3 theta!=0, J=20, de=5,  kCR2=0  ", 20.0, 5.0, 0.0),
                           ("D4 theta!=0, J=20, de=5,  kCR2=2e6", 20.0, 5.0, 2e6),
                           ("D5 theta!=0, J=20, de=0,  kCR2=2e6", 20.0, 0.0, 2e6)]:
    ts.hams = mk(J, de)
    odd_test(label + " (J, de in mT)", **dict(base, kCR2=kCR2))
ts.hams = mk(20.0, 0.0)
b = dict(base); b.update(Pb=PS, PCR=PS, PCR2=PS)
odd_test("D6 theta=0 (no CISS, no SOC shift), J=20, kCR2=2e6", **dict(b, kCR2=2e6))

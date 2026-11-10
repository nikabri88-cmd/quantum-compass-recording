import numpy as np
import two_site as ts
from two_site import *
base = dict(k2=3e7, km2=4e6, kCR=1.5e7, kSP=1e6)
noc = dict(Pb=PS, PCR=PS, PCR2=PS, U2=ONE)
print("All tests below: NO CISS anywhere (singlet birth at the bridge, identity hop, singlet recombination).")
b = dict(base); b["kCR"] = 0.0
odd_test("T9  recombination only from the terminal site", kCR2=2e6, **noc, **b)
for k2 in (3e8, 3e9, 3e10):
    b = dict(base); b["k2"] = k2
    odd_test(f"T10 both channels, forward hop k2 = {k2:.0e}", kCR2=2e6, **noc, **b)
b = dict(base); b["k2"] = b["km2"] = 1e7
odd_test("T13 composite, fast exchange both ways, recomb. only at birth site", kCR2=0.0, **noc, **b)
odd_test("T13b same, recombination at both sites", kCR2=2e6, **noc, **b)

# identical Hamiltonians at both sites
orig = ts.hams
def same_h(bv, **kw):
    Hb, Hcs = orig(bv, **kw)
    return Hcs, Hcs
ts.hams = same_h
odd_test("T12 both channels, identical spin Hamiltonians at both sites", kCR2=2e6, **noc, **base)

# bridge differs from terminal site only by exchange (same nuclei, same dipolar)
def exch_only_diff(bv, **kw):
    Hb, Hcs = orig(bv, **kw)
    return Hcs - GAM * 2 * 0.3 * sum(SA[i] @ SB[i] for i in range(3)), Hcs
ts.hams = exch_only_diff
odd_test("T12b bridge = terminal Hamiltonian + exchange 0.3 mT", kCR2=2e6, **noc, **base)

# bridge: FAD nucleus + exchange only (no W_B nucleus, no dipolar); terminal: full
def bridge_fad_only(bv, B0=0.05, **kw):
    Hb, Hcs = orig(bv, B0=B0, **kw)
    Zee = GAM * B0 * (dot(bv, SA) + dot(bv, SB))
    Hb2 = Zee + GAM * bil(ts.A1, SA, I1) - GAM * 2 * 0.3 * sum(SA[i] @ SB[i] for i in range(3))
    return Hb2, Hcs
ts.hams = bridge_fad_only
odd_test("T11 bridge: FAD nucleus + exchange only; terminal: full", kCR2=2e6, **noc, **base)

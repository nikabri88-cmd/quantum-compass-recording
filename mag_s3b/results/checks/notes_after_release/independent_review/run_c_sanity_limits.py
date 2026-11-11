import numpy as np
import two_site as ts
from two_site import *
noc = dict(Pb=PS, PCR=PS, PCR2=PS, U2=ONE)
orig = ts.hams
def same_h(bv, **kw):
    Hb, Hcs = orig(bv, **kw)
    return Hcs, Hcs
ts.hams = same_h
print("No CISS. Sanity limits that must be exactly even:")
odd_test("T12c identical Hamiltonians and identical recombination rates", k2=3e7, km2=4e6, kCR=2e6, kCR2=2e6, kSP=1e6, **noc)
ts.hams = orig
odd_test("T2c  different Hamiltonians, no product channel asymmetry: kCR2 = 0", k2=3e7, km2=4e6, kCR=1.5e7, kCR2=0.0, kSP=1e6, **noc)

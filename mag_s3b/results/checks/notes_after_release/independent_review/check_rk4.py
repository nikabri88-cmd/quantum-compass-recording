"""Independent check of the no-CISS odd part (case T2) without superoperators: RK4 on the two block density
matrices written directly with commutators and anticommutators, compared with the linear solve of two_site.py."""
import numpy as np
from two_site import *

k2, km2, kCR, kCR2, kSP = 3e7, 4e6, 1.5e7, 2e6, 1e6


def rhs(rb, rc, Hb, Hcs):
    db = -1j * (Hb @ rb - rb @ Hb) - 0.5 * kCR * (PS @ rb + rb @ PS) - k2 * rb + km2 * rc
    dc = -1j * (Hcs @ rc - rc @ Hcs) - 0.5 * kCR2 * (PS @ rc + rc @ PS) - (km2 + kSP) * rc + k2 * rb
    return db, dc


def run(b, T=5e-6, dt=2.5e-11):
    Hb, Hcs = hams(b)
    rb, rc = PS / Z, np.zeros_like(PS)
    n = int(round(T / dt))
    acc = np.zeros(3)
    def obs(rb, rc):
        return np.array([kSP * np.trace(rc).real, kCR * np.trace(PS @ rb).real, kCR2 * np.trace(PS @ rc).real])
    prev = obs(rb, rc)
    for _ in range(n):
        a1, c1 = rhs(rb, rc, Hb, Hcs)
        a2, c2 = rhs(rb + 0.5 * dt * a1, rc + 0.5 * dt * c1, Hb, Hcs)
        a3, c3 = rhs(rb + 0.5 * dt * a2, rc + 0.5 * dt * c2, Hb, Hcs)
        a4, c4 = rhs(rb + dt * a3, rc + dt * c3, Hb, Hcs)
        rb = rb + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        rc = rc + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
        cur = obs(rb, rc)
        acc += 0.5 * (prev + cur) * dt
        prev = cur
    left = (np.trace(rb) + np.trace(rc)).real
    return acc, left


v = np.array([0.36, -0.48, 0.80])
for name, d in (("+v", v), ("-v", -v)):
    acc, left = run(d)
    ref = np.array(yields(d, Pb=PS, PCR=PS, PCR2=PS, U2=ONE, k2=k2, km2=km2, kCR=kCR, kCR2=kCR2, kSP=kSP))
    print(f"{name}: RK4  Phi_SP {acc[0]:.9f} Phi_CR {acc[1]:.9f} Phi_CR2 {acc[2]:.9f}  (population left {left:.1e})")
    print(f"{name}: solve Phi_SP {ref[0]:.9f} Phi_CR {ref[1]:.9f} Phi_CR2 {ref[2]:.9f}")
    if name == "+v":
        ap, rp = acc, ref
    else:
        print(f"odd part of Phi_SP: RK4 {(ap[0] - acc[0]) / 2:+.3e}   solve {(rp[0] - ref[0]) / 2:+.3e};"
              f"   odd part of Phi_CR: RK4 {(ap[1] - acc[1]) / 2:+.1e}  solve {(rp[1] - ref[1]) / 2:+.1e}")

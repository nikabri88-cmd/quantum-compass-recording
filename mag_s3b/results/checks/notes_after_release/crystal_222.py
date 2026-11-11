"""Note 4 of v0.2.1: the odd part of the yield in a crystal of point group 222 (6PU0 crystallises in P2_1 2_1 2_1).

The four molecules of the cell are related by rotations by 180 degrees about the three cell axes. If all four are
excited alike, the yield of the crystal is the mean of the molecular yield over these rotations. Its odd part is then
odd under each of the three mirrors of the cell, so it vanishes for every field in a cell plane, has no dipole (l = 1)
part, and its lowest pattern is xyz. Checked on a toy pair born in form K at m = 1 (its own odd part is nonzero).
Pass/fail and residuals only.
"""
import numpy as np
from toy_pair import form_K, yield_F, unit, random_directions

rho_K = form_K(0.1 * np.pi, unit([0.3, -0.5, 0.8]))
ROT = [np.diag(d) for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
MIRROR = [np.diag(d) for d in ((-1, 1, 1), (1, -1, 1), (1, 1, -1))]


def molecule(B):
    return yield_F(B, rho_K)


def crystal(B):
    return np.mean([molecule(R @ B) for R in ROT])


def odd(f, B):
    return (f(B) - f(-B)) / 2


dirs = random_directions(4, seed=11)
scale = max(abs(odd(molecule, B)) for B in dirs)
print(f"single molecule: odd part nonzero = {scale > 1e-9}")

worst_plane = 0.0
for axis in range(3):
    for B in dirs:
        Bp = B.copy()
        Bp[axis] = 0.0
        worst_plane = max(worst_plane, abs(odd(crystal, unit(Bp))))
print(f"crystal, fields in the three cell planes: worst |odd| / molecular odd scale = {worst_plane / scale:.1e}")

worst_mirror = max(abs(odd(crystal, S @ B) + odd(crystal, B)) for S in MIRROR for B in dirs)
print(f"crystal, odd part changes sign under each cell mirror: worst residual / molecular odd scale = "
      f"{worst_mirror / scale:.1e}")

off = min(abs(odd(crystal, B)) for B in dirs)
print(f"crystal, fields off the cell planes: odd part nonzero = {off / scale > 1e-6}")

"""
Lattice geometry utilities.

Builds neighbour lists for 1D chains and 2D square lattices with
periodic boundary conditions (PBC). Used by Hamiltonian builders
and ansatz input-encoding layers.
"""
from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np


# ---------- 1D chain ----------

def chain_neighbours(
    L: int,
    pbc: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Nearest-neighbour (nn) and next-nearest-neighbour (nnn) pairs
    for a 1D chain of length L.

    Args:
        L: Number of sites.
        pbc: If True, add periodic boundary bonds.

    Returns:
        (nn_pairs, nnn_pairs) — each a list of (i, j) tuples with i < j.
    """
    nn = [(i, (i + 1) % L) for i in range(L)]
    nnn = [(i, (i + 2) % L) for i in range(L)]

    if not pbc:
        nn = [(i, j) for i, j in nn if j > i]
        nnn = [(i, j) for i, j in nnn if j > i]

    # Normalise: ensure i < j and remove duplicates
    nn = _normalise_pairs(nn)
    nnn = _normalise_pairs(nnn)
    return nn, nnn


# ---------- 2D square lattice ----------

def square_neighbours(
    Lx: int,
    Ly: int,
    pbc: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Nearest-neighbour and next-nearest-neighbour pairs for a 2D
    square lattice of size Lx × Ly.

    Site indexing: site = y * Lx + x  (row-major).

    Args:
        Lx: Number of columns.
        Ly: Number of rows.
        pbc: If True, add periodic boundary bonds.

    Returns:
        (nn_pairs, nnn_pairs) — each a list of (i, j) tuples with i < j.
    """
    def idx(x: int, y: int) -> int:
        return y * Lx + x

    nn: list[Tuple[int, int]] = []
    nnn: list[Tuple[int, int]] = []

    for y, x in itertools.product(range(Ly), range(Lx)):
        site = idx(x, y)

        # Nearest neighbours: right and down
        # Right
        nx = (x + 1) % Lx
        if pbc or nx > x:
            nn.append((site, idx(nx, y)))

        # Down
        ny = (y + 1) % Ly
        if pbc or ny > y:
            nn.append((site, idx(x, ny)))

        # Next-nearest neighbours: down-right and down-left diagonals
        # Down-right
        nx_dr, ny_dr = (x + 1) % Lx, (y + 1) % Ly
        if pbc or (nx_dr > x and ny_dr > y):
            nnn.append((site, idx(nx_dr, ny_dr)))

        # Down-left
        nx_dl, ny_dl = (x - 1) % Lx, (y + 1) % Ly
        if pbc or (nx_dl < x and ny_dl > y):
            nnn.append((site, idx(nx_dl, ny_dl)))

    nn = _normalise_pairs(nn)
    nnn = _normalise_pairs(nnn)
    return nn, nnn


def site_coords(Lx: int, Ly: int) -> np.ndarray:
    """Return (N, 2) array of (x, y) coordinates for each site."""
    coords = np.empty((Lx * Ly, 2), dtype=np.float64)
    for y in range(Ly):
        for x in range(Lx):
            coords[y * Lx + x] = (x, y)
    return coords


# ---------- Internal helpers ----------

def _normalise_pairs(
    pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Ensure i < j in each pair and remove duplicates."""
    seen: set[Tuple[int, int]] = set()
    result: list[Tuple[int, int]] = []
    for i, j in pairs:
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return sorted(result)

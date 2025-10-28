#!/usr/bin/env python
"""An individual file made to test how numpy array gen will perform on a script sent into the taskfarm."""

import numpy as np
import sys
# from config import N


def generate_phi(num: int = 0) -> np.ndarray:
    phi = np.random.uniform(0, 2 * np.pi, (num, 4))

    return phi


def generate_t(num: int = 0) -> np.ndarray:
    t = np.random.uniform(0, 1, (num, 5))
    return t


def generate_r(t: np.ndarray) -> np.ndarray:
    t1, t2, t3, t4, t5 = t.T
    r1 = np.sqrt(1 - t1**2)
    r2 = np.sqrt(1 - t2**2)
    r3 = np.sqrt(1 - t3**2)
    r4 = np.sqrt(1 - t4**2)
    r5 = np.sqrt(1 - t5**2)
    return np.stack([r1, r2, r3, r4, r5], axis=1)


def calculate_t_prime(t: np.ndarray, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    r1, r2, r3, r4, r5 = r.T
    t1, t2, t3, t4, t5 = t.T
    phi1, phi2, phi3, phi4 = phi.T

    numerator = (
        -(np.exp(1j * (phi1 + phi4 - phi2)) * (r1 * r3 * r5 * t2 * t4))
        + ((t2 * t4) * (np.exp(1j * (phi1 + phi4))))
        - (np.exp(1j * phi4) * t1 * t3 * t4)
        + (np.exp(1j * phi3) * r2 * r3 * r4 * t1 * t5)
        - (np.exp(1j * phi1) * t2 * t3 * t5)
    )
    denominator = (
        -1
        - (r2 * r3 * r4 * np.exp(1j * (phi3)))
        + (r1 * r3 * r5 * np.exp(1j * phi2))
        + (r1 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3)))
        + (t1 * t2 * t3 * np.exp(1j * phi1))
        - (t1 * t2 * t4 * t5 * np.exp(1j * (phi1 + phi4)))
        + (t3 * t4 * t5 * np.exp(1j * phi4))
    )

    return np.abs(np.abs(numerator) / np.abs(denominator))


if __name__ == "__main__":
    N = int(sys.argv[1])
    output_dir = str(sys.argv[2])
    if not N:
        N = 1000000
    t = generate_t(N)
    r = generate_r(t)
    phi = generate_phi(N)
    t_prime = calculate_t_prime(t, r, phi)

    filename = f"{output_dir}/data/t_prime_{N}_samples"
    np.savetxt(filename, t_prime)
    print(f"Results saved to {filename}.")

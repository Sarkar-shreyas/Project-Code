"""A file to test how a beta distribution fits to a fixed point distribution of t'"""

import numpy as np
from typing import Optional
from scipy.stats import beta, chi2_contingency
from src.distribution_production import Probability_Distribution
from config import N, T_RANGE, BINS, EXPRESSION
import matplotlib.pyplot as plt
from time import time


def load_data(filename: Optional[str] = None) -> np.ndarray:
    """A function to load the data from the given filename into a numpy array for later use."""
    if not filename:
        filename = "data/Shaw/Fixed_point/Converged_t_prime_80000000_iters.txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = np.loadtxt(file, encoding="utf-8")
    except Exception as e:
        raise e

    return data


def plot_beta_derived_params(data: np.ndarray, start_time) -> np.ndarray:
    """A function to perform a beta fit to the input data and keep track of the time taken. Fits entirely without an input loc and scale."""
    print("Beginning fitting without constraints")
    alpha, b, loc, scale = beta.fit(data)
    print(f"Fitting done in {time() - start_time:.3f} seconds.")
    print(f"Fitted with values: Alpha={alpha}, Beta={b}, Location={loc}, Scale={scale}")
    beta_samples = beta.rvs(alpha, b, size=N)

    values, bins = np.histogram(beta_samples, bins=BINS, range=T_RANGE, density=True)
    if min(values) == 0:
        values += 1e-3

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, values, label="Beta derived params")
    print("Plotted data for derived params.")
    return values


def plot_beta_fixed_params(
    data: np.ndarray, location: int, scale: int, start_time
) -> np.ndarray:
    """A function to perform a beta fit to the input data with a fixed location and scale. Also keeps track of the time taken."""
    print(f"Beginning fitting with fixed Location={location} and Scale={scale}")
    alpha, b, loc, scale = beta.fit(data, floc=location, fscale=scale)
    beta_samples = beta.rvs(alpha, b, size=N)
    print(f"Fitting done in {time() - start_time:.3f} seconds.")
    print(f"Fitted with values: Alpha={alpha}, Beta={b}, Location={loc}, Scale={scale}")
    values, bins = np.histogram(beta_samples, bins=BINS, range=T_RANGE, density=True)
    if min(values) == 0:
        values += 1e-3

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, 3 * values, label="Beta fixed params")
    print("Plotted data for fixed constraints")
    return values


# TODO: Fix later if time permits, currently wrong implementation.
def chi_square(
    original_data: np.ndarray,
    beta_sample_derived_params: np.ndarray,
    beta_sample_fixed_params: np.ndarray,
) -> dict:
    """A function to perform a chi-square goodness of fit test to the beta distributions fitted to assess which would be more suitable."""
    derived_params_table = np.stack([original_data, beta_sample_derived_params], axis=1)
    fixed_params_table = np.stack([original_data, beta_sample_fixed_params], axis=1)
    stat1, p1, dof1, expected1 = chi2_contingency(derived_params_table)
    stat2, p2, dof2, expected2 = chi2_contingency(fixed_params_table)
    return {
        "derived": {"stats": stat1, "p": p1},
        "fixed": {"stats": stat2, "p": p2},
    }


if __name__ == "__main__":
    start_time = time()
    # t_prime_data = load_data("data/Shaw/Fixed_point/t_prime_1000000_iters.txt")

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #
    print("Starting to load data")
    print("-" * 100)
    # ----------------------------------------------------------Loading Data---------------------------------------------------------- #

    t_prime_data = load_data(f"data/{EXPRESSION}/Fixed_point/t_prime_{N}_iters.txt")

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #
    print(f"Data was loaded {time() - start_time:.3f} seconds after beginning.")
    print("-" * 100)
    # -----------------------------------------------------Plotting original data----------------------------------------------------- #

    dist = Probability_Distribution(t_prime_data, BINS, T_RANGE)
    bin_centers = 0.5 * (dist.bin_edges[1:] + dist.bin_edges[:-1])
    plt.plot(
        bin_centers, dist.histogram_values, label="Original data", linestyle="dashed"
    )

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #
    print(
        f"Distribution was created {time() - start_time:.3f} seconds after beginning."
    )
    print("-" * 100)
    print("Beginning beta fitting")
    print("=" * 100)
    # -----------------------------------------------------Plotting derived data------------------------------------------------------ #

    derived_data = plot_beta_derived_params(dist.raw_values, start_time)

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #
    print(
        f"The beta distribution has been fitted without constraints {time() - start_time:.3f} seconds after beginning."
    )
    print("-" * 100)
    # ------------------------------------------------------Plotting fixed data------------------------------------------------------- #

    fixed_data = plot_beta_fixed_params(dist.raw_values, 0, 1, start_time)

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #
    print(
        f"The beta distribution has been fitted with constraints {time() - start_time:.3f} seconds after beginning."
    )
    print("=" * 100)
    # ----------------------------------------------------------Saving Plot----------------------------------------------------------- #

    plt.legend()
    plt.title("Hist values of base distribution and beta samples vs domain")
    plt.savefig(f"test_plots/{EXPRESSION}/test_beta_{N}_iters.png", dpi=150)

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #

    print(f"Figures have been plotted in {time() - start_time:.3f} seconds.")
    print("Beginning Chi-Square analysis")
    print("=" * 100)

    # --------------------------------------------------------Chi Squared Test-------------------------------------------------------- #

    # results = chi_square(dist.histogram_values, derived_data, fixed_data)

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #

    # print(f"Chi squared test finished in {time() - start_time:.3f} seconds.")
    # print(results)
    # print("=" * 100)

    # ------------------------------------------------------------PRINTING------------------------------------------------------------ #
    print(f"The program has finished in {time() - start_time:.3f} seconds.")
    print("-" * 100)

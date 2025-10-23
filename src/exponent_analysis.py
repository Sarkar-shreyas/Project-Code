import numpy as np
from .utils import savedata
from .rg_iterator import rg_iterator_for_nu
from .distribution_production import Probability_Distribution
from config import N, K, Z_RANGE, BINS, EXPRESSION
from scipy.stats import norm, linregress
import sys

# from scipy.optimize import least_squares
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Peak finders ---------- #
def get_peak_from_subset(z_subset: np.ndarray) -> float:
    """Estimate the peak location of a Q(z) distribution subset using Shaw's method.

    The function builds a histogram of the provided subset, identifies the histogram
    bin with the maximum density, and grows a symmetric interval around that bin
    until approximately 5% of the total probability mass (in that neighbourhood) is
    captured. It then selects the raw z values that fall inside that interval and
    fits a Gaussian to those "tip" values using ``scipy.stats.norm.fit``. The
    returned value is the absolute value of the fitted mean (mu). If the fit
    produces a non-finite mean, the function falls back to a weighted average of
    the bin centres inside the selected interval.

    Parameters
    ----------
    z_subset : numpy.ndarray
        One-dimensional array of sampled z values representing a subset of the
        full Q(z) distribution. The array may be empty or contain very few
        elements; the function attempts to handle these cases gracefully.

    Returns
    -------
    float
        Estimated peak location (absolute value) for the subset. If the fit
        fails or produces non-finite results, a weighted average of bin centres
        is returned instead.
    """
    # Set up the histogram from the input subset
    z_values, bin_edges = np.histogram(z_subset, bins=BINS, range=Z_RANGE, density=True)

    # z-mass per bin
    bin_widths = np.diff(bin_edges)
    z_masses_per_bin = bin_widths * z_values

    # Find the indexes that would sort the bins, without sorting in place
    max_index = int(np.argmax(z_values))
    left_index = max_index
    right_index = max_index
    # print(z_values[max_index])
    # print(z_masses_per_bin[max_index])

    # Store the cumulative sum to check whether we've hit 5%, initialise at the max
    cumulative_z_sum = z_masses_per_bin[max_index]
    # print(z_masses_per_bin)
    # print(z_masses_per_bin.mean())
    # print(z_values.sum())
    # Grow around the maximum value until 5% of probability mass is stored
    while cumulative_z_sum < 0.05 and (
        left_index > 0 or right_index < len(z_values) - 1
    ):
        # Set conditional to move left
        go_left = (right_index >= len(z_values) - 1) or (
            left_index > 0 and z_values[left_index - 1] >= z_values[right_index + 1]
        )
        # Go along the higher direction
        if go_left:
            left_index -= 1
            cumulative_z_sum += z_masses_per_bin[left_index]
        else:
            right_index += 1
            cumulative_z_sum += z_masses_per_bin[right_index]

    # Setup the slicing bounds for the z array
    z_low = bin_edges[left_index]
    z_high = bin_edges[right_index + 1]

    # Use values from the raw subset, not histogram data
    z_tip_values = z_subset[(z_subset >= z_low) & (z_subset < z_high)]
    # x = np.linspace(z_tip_values.min(), z_tip_values.max(), len(z_tip_values))
    # print(z_tip_values)
    # print(z_values[left_index : right_index + 1])
    # print(left_index, right_index)
    # plt.close()
    # plt.plot(x, z_tip_values, label="z_tip vs bins")
    # plt.plot(
    #     bin_edges[left_index : right_index + 1],
    #     z_values[left_index : right_index + 1],
    #     label="z_subset vs bins",
    # )
    # plt.legend()
    # plt.savefig("test_plot.png", dpi=300)
    # sys.exit(0)
    # Use scipy's norm fit to apply a gaussian fit
    mu, _ = norm.fit(z_tip_values)
    # print(mu, _)
    # sys.exit(0)
    # return float(mu)
    # Prevent infinite values messing up the log
    if not np.isfinite(mu):
        bin_values = z_values[left_index : right_index + 1]
        mu = float(
            np.average(
                bin_values,
                weights=z_masses_per_bin[left_index : right_index + 1],
            )
        )
    return float(np.abs(mu))


def estimate_z_peak(Q_z: Probability_Distribution) -> float:
    """Estimate the average peak location for a full sample by aggregating subset peaks.

    The input sample is split into 10 equal (or near-equal) subsets. For each
    subset the ``get_peak_from_subset`` function is used to estimate a local
    peak. The arithmetic mean of these per-subset peak estimates is returned.

    Parameters
    ----------
    z_sample : numpy.ndarray
        One-dimensional array of sampled z values from a Probability_Distribution
        object. The array should contain enough samples to be split into the
        default 10 subsets; if it contains fewer elements, some subsets will be
        empty and ``get_peak_from_subset`` will handle them.

    Returns
    -------
    float
        Arithmetic mean of the per-subset peak estimates.
    """
    z_subsets = np.array([Q_z.sample(N // 10) for _ in range(10)])

    mu_values = [get_peak_from_subset(z_subset) for z_subset in z_subsets]

    return float(np.mean(mu_values))


# ---------- Fitting helper ---------- #
def fit_z_peaks(x: np.ndarray, y: np.ndarray, method: str = "ls") -> tuple:
    if method == "ls":
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x_mean - x) * (y_mean - y)) / np.sum(
            (x_mean - x) * (x_mean - x)
        )
        intercept = y_mean - slope * x_mean
        residual = y - slope * x - intercept
        ssr = float(np.dot(residual, residual))
        sst = float(np.dot(y - y_mean, y - y_mean))
        r2 = 1 - (ssr / sst)
        return float(np.abs(slope)), float(r2)

    elif method == "linear":
        result = linregress(x, y)
        slope = result.slope  # type: ignore
        r2 = result.rvalue**2  # type: ignore
        return float(np.abs(slope)), float(r2)
    elif method == "Levenberg":
        pass

    return ()


# ---------- Nu calculator ---------- #
def calculate_nu(slope: float, rg_steps: int = K) -> float:
    """Calculates the critical exponent nu based on the input slope and number of RG steps performed, k"""
    nu = rg_steps * (np.log(2) / np.abs(np.log(np.abs(slope))))
    return nu


# ---------- Critical Exponent estimation factory ---------- #
def critical_exponent_estimation(
    fixed_point_Qz: Probability_Distribution,
) -> dict:
    """Estimate critical exponent nu using RG flow analysis of perturbed distributions.

    This function implements a multi-step analysis to estimate the critical exponent nu:
    1. Applies a series of small perturbations to a fixed-point distribution Q(z)
    2. For each perturbation:
       - Tracks the evolution of distribution peaks through K RG steps
       - Uses ``estimate_z_peak`` to locate peaks in perturbed distributions
    3. Performs linear regression between initial perturbations and evolved peaks
    4. Estimates nu using the scaling relation nu = ln(2^k)/ln(z_k/z_0)

    The analysis includes visualization of the RG flow for z_0 = 0.007 and tracks
    computation time for each major step. A figure showing the evolution of Q(z)
    is saved to the plots directory.

    Parameters
    ----------
    fixed_point_Qz : Probability_Distribution
        The fixed-point distribution Q*(z) around which to perform perturbative
        analysis. This distribution should be at or very near the RG fixed point.

    Returns
    -------
    dict
        A dictionary containing:
        - 'Nu_values': List of nu estimates for each RG step
        - 'Nu_data': Dict with mean/median nu values and analysis bounds
        - 'parameters': List of dicts with slope and R² for each RG step
        - 'z_peaks': 2D array of peak locations [RG_step, perturbation]
        - 'perturbations': List of perturbation magnitudes used

    Notes
    -----
    The function uses a predefined set of perturbations from 1e-3 to K*1e-3
    and averages nu estimates between RG steps 5 and 12 for the final result.
    Progress and timing information is printed to stdout during execution.
    """
    # Set up list of perturbations to try
    perturbation_list = np.linspace(0, 1e-3 * K, K + 1)
    num_perturbations = len(perturbation_list)

    # Set up an empty array to track z peaks
    z_peaks = np.zeros((K + 1, num_perturbations)).astype(float)
    unperturbed_z_peak = estimate_z_peak(fixed_point_Qz)

    # Setup graphs
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 4))
    ax0.set_xlim(Z_RANGE)
    ax0.set_ylim([0, 0.3])
    ax0.set_xlabel("z")
    ax0.set_ylabel("Q(z)")
    ax0.set_title("Q(z) vs z with z_0 = 0.007")

    # Plot unperturbed distribution
    unperturbed_centers = 0.5 * (
        fixed_point_Qz.bin_edges[:-1] + fixed_point_Qz.bin_edges[1:]
    )
    ax0.plot(unperturbed_centers, fixed_point_Qz.histogram_values, label="Unperturbed")

    ax1.set_xlim([0, max(perturbation_list)])
    ax1.set_ylim([0, 0.1])
    ax1.set_xlabel("z_0")
    ax1.set_ylabel("z_peak")
    ax1.set_title("z_peak vs z_0")

    print("-" * 100)
    print("Beginning z peak calculations")
    start_time = time.time()

    # Run through the perturbations in the list
    for i, perturbation in enumerate(perturbation_list):
        z_sample = fixed_point_Qz.sample(N)
        perturbed_z = z_sample - perturbation

        # Generates the perturbed distribution
        perturbed_Qz = Probability_Distribution(perturbed_z, BINS)

        # Store the first peak prior to any RG steps for each perturbation
        z_peaks[0, i] = estimate_z_peak(perturbed_Qz) - unperturbed_z_peak
        # z_peaks[0, i] = estimate_z_peak(perturbed_Qz.sample(N))

        print(f"Performing RG step on perturbation {i}, z_0 = {perturbation:.3f}")

        # Perform RG iterations for the specific perturbation
        for n in range(1, K + 1):
            next_Qz = rg_iterator_for_nu(perturbed_Qz)
            # next_z_sample = next_Qz.sample(N)

            # Store the difference between the peak of the perturbed distribution and the unperturbed distribution
            # z_peaks[n, i] = estimate_z_peak(next_z_sample) - unperturbed_z_peak
            z_peaks[n, i] = estimate_z_peak(next_Qz) - z_peaks[n - 1, i]
            print(f"RG Step #{n} done for perturbation {i}")
            print(
                f"Time elapsed since analysis began: {time.time() - start_time:.3f} seconds"
            )
            if perturbation == 7e-3:
                centers = 0.5 * (next_Qz.bin_edges[:-1] + next_Qz.bin_edges[1:])
                ax0.plot(centers, next_Qz.histogram_values, label=f"RG{n}")

            perturbed_Qz = next_Qz
        print(
            f"All RG steps done for perturbation {i}. Time elapsed: {time.time() - start_time:.3f} seconds since beginning z peak calculations"
        )
        print("-" * 100)

    # Plot z_peaks against z_0
    for i in range(K + 1):
        # if i % 3 == 1:
        peaks_over_size: np.ndarray = z_peaks[i, :]
        slope, r2 = fit_z_peaks(perturbation_list, peaks_over_size, "linear")
        ax1.plot(perturbation_list, slope * perturbation_list, label=f"RG step{i}")
        # ax1.plot(perturbation_list, peaks_over_size, label=f"RG step {i}")
        savedata(z_peaks[i, :].tolist(), f"Perturbed Dist and z_n/z_peaks_{N}_iters")
        savedata(
            peaks_over_size.tolist(),
            f"Perturbed Dist and z_n/z_peaks_over_system_size_{N}_iters",
        )

    # Get the plots saved
    ax0.legend(loc="upper left")
    ax1.legend(bbox_to_anchor=(1.02, 1))
    plt.savefig(
        f"plots/{EXPRESSION}/Perturbed Dist and z_n/Q(z)_perturbed_by_0.007_with_{N}_iters.png",
        dpi=150,
    )
    plt.close()
    print("-" * 100)
    print(
        f"z peaks have been found. Time elapsed to complete calculations: {time.time() - start_time:.3f}"
    )
    print("Starting linear regression analysis")
    print("=" * 100)
    current_time = time.time()
    print(
        f"Analysis starting {current_time - start_time:.3f} seconds after beginning calculations"
    )
    # Find the estimation of nu for each perturbation and RG step taken
    nu_estimates = []
    params = []

    plt.xlabel("2^n")
    plt.ylabel("nu")

    for n in range(K + 1):
        print(
            f"Performing Nu estimation for RG step #{n} {time.time() - current_time:.3f} seconds after beginning analysis."
        )
        y = z_peaks[n, :].astype(float)
        x = np.array(perturbation_list).astype(float)

        # Slice x and y values to avoid infinite values
        # mask = np.isfinite(x) & np.isfinite(y)
        # x = x[mask]
        # y = y[mask]

        # Fit y against x using the default least squares definition
        absolute_slope, r2 = fit_z_peaks(x, y)
        nu = float(calculate_nu(absolute_slope, n))
        nu_estimates.append(nu)
        params.append(
            {"RG step no.": n, "Slope": float(absolute_slope), "R2": float(r2)}
        )

    system_size = [2**i for i in range(K + 1)]
    plt.xlim([0, 2**K])
    plt.ylim([0, max(nu_estimates) + 1])
    plt.scatter(system_size, nu_estimates)
    plt.title("Nu from absolute slopes against system size 2^n")
    plt.savefig(
        f"plots/{EXPRESSION}/Nu/Nu_against_system_size_for_{N}_iters.png", dpi=150
    )
    print("=" * 100)
    print(
        f"Analysis completed after {time.time() - current_time:.3f} seconds, returning results"
    )
    return {
        "Nu": nu_estimates,
        "parameters": params,
        "z_peaks": z_peaks.tolist(),
        "perturbations": perturbation_list.tolist(),
    }

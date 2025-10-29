"""Tools for generating and manipulating probability distributions in RG analysis.

This module provides functionality for creating initial distributions, sampling from
them, and performing various distribution manipulations required by the RG flow
analysis. It includes tools for phase generation, distribution centering, and
maintaining probability distribution invariants.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, beta
from config import N, T_RANGE, Z_RANGE, BINS
from time import time
import sys


# ---------- Initial distribution helpers ---------- #
def generate_random_phases(N: int) -> np.ndarray:
    """Generate random phase angles for RG transformation.

    Creates an array of uniformly distributed random phases in [0, 2π]
    used in the RG transformation step.

    Parameters
    ----------
    N : int
        Number of phase sets to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 4) containing random phases in [0, 2π].
    """
    phi_sample = np.random.uniform(0, 2 * np.pi, (N, 4))
    return phi_sample


def generate_initial_t_distribution(N: int) -> np.ndarray:
    """Generate initial amplitude distribution P(t).

    Creates an initial distribution of amplitudes t with the property that
    the squared amplitudes g = |t|² are uniformly distributed in [0,1].
    This ensures P(t) is symmetric about t² = 0.5.

    Parameters
    ----------
    N : int
        Number of samples to generate.

    Returns
    -------
    numpy.ndarray
        Array of N amplitude values t = √g where g ~ U[0,1].
    """
    # g_sample = np.random.uniform(0, 1.0 + 1e-15, N)
    g_sample = np.random.uniform(0, 1.0, N)
    t_dist = np.sqrt(g_sample)
    return t_dist


# ---------- Sampling helpers ---------- #
class Probability_Distribution:
    """A class for managing binned probability distributions in the RG analysis.

    This class provides tools for creating, analyzing, and sampling from
    probability distributions represented as normalized histograms. It supports
    operations needed for the RG flow analysis including distance calculations
    between distributions and statistical measurements.

    Parameters
    ----------
    values : numpy.ndarray
        Raw values to bin into a probability distribution.
    bins : int, optional
        Number of bins for the histogram (default from config.BINS).
    range : tuple, optional
        (min, max) range for binning (default from config.Z_RANGE).

    Attributes
    ----------
    bin_edges : numpy.ndarray
        Edges of the histogram bins, shape (bins+1,).
    histogram_values : numpy.ndarray
        Normalized histogram values, shape (bins,).
    cdf : numpy.ndarray
        Cumulative distribution function, shape (bins,).
    """

    def __init__(self, values, bins=BINS, range=Z_RANGE):
        histogram_values, bin_edges = np.histogram(
            values, bins=bins, range=range, density=True
        )
        cdf = histogram_values.cumsum()
        cdf = cdf / cdf[-1]
        self.domain_min, self.domain_max = range
        self.domain_width = np.abs(self.domain_min) + np.abs(self.domain_max)
        self.raw_values = values
        self.bin_edges = bin_edges
        self.histogram_values = histogram_values
        self.cdf = cdf

    def histogram_distances(
        self, other_histogram_values, other_histogram_bin_edges
    ) -> float:
        """Calculate L2 distance between this distribution and another.

        Computes the integrated squared difference between two normalized
        histograms: δ = √∫(Q_{k+1}(z)² - Q_k(z)²)dz.

        Parameters
        ----------
        other_histogram_values : numpy.ndarray
            Normalized values from other histogram, shape (bins,).
        other_histogram_bin_edges : numpy.ndarray
            Bin edges from other histogram, shape (bins+1,).

        Returns
        -------
        float
            L2 distance between the distributions.
        """
        integrand = (self.histogram_values - other_histogram_values) ** 2
        dz = np.diff(other_histogram_bin_edges)
        distance = float(np.sqrt(np.sum(integrand * dz)))
        return distance

    def update(self, new_hist_values, new_bin_edges):
        """Replace the stored histogram and recompute the CDF.

        Parameters
        ----------
        new_hist_values : numpy.ndarray
            Normalized histogram values for each bin (shape: (bins,)). The
            values should represent a probability density over the bin widths.
        new_bin_edges : numpy.ndarray
            Bin edges corresponding to the histogram (shape: (bins+1,)).

        Notes
        -----
        This method updates the distribution in-place and recomputes the
        cumulative distribution function (CDF) from the provided histogram
        values. If the provided histogram values are not normalized, the
        computed CDF will be scaled by their cumulative sum (i.e. cdf[-1]).
        The caller should ensure that the inputs are valid (non-negative and
        consistent shapes) to avoid runtime errors.
        """
        self.histogram_values = new_hist_values
        self.bin_edges = new_bin_edges
        cdf = new_hist_values.cumsum()
        cdf /= cdf[-1]
        self.cdf = cdf

    def mean_and_std_from_hist(self) -> tuple:
        """Calculate mean and standard deviation of the distribution.

        Uses the normalized histogram values to compute first and second
        moments of the distribution.

        Returns
        -------
        tuple
            (mean, standard_deviation) of the distribution.
        """
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        mean = (centers * self.histogram_values).sum() / self.histogram_values.sum()
        variance = (
            ((centers - mean) ** 2) * self.histogram_values
        ).sum() / self.histogram_values.sum()
        standard_deviation = np.sqrt(variance)
        return mean, standard_deviation

    def mean_and_std_from_raw_data(self) -> tuple:
        mean = np.mean(self.raw_values)
        std = np.std(self.raw_values)
        return mean, std

    # TODO: Sort out sampler for both t and z distributions, decide on how to handle them
    def sample_t(self, num: int) -> np.ndarray:
        """Generate random samples from the distribution.

        Uses inverse transform sampling: generates uniform random numbers,
        maps them through the CDF, and interpolates within bins to get
        continuous values.

        Parameters
        ----------
        N : int
            Number of samples to generate.

        Returns
        -------
        numpy.ndarray
            Array of N samples drawn from the distribution.
        """

        # Fit the data to a beta distribution
        alpha, b, _, _ = beta.fit(self.raw_values, floc=0, fscale=1)
        # print("Fitting done.")
        # Construct the PDF q(t)
        fitted_distribution = beta.pdf(self.raw_values, alpha, b, loc=0, scale=1)

        # Get the indices of the bins each value falls under, then we can slice out the relevant values
        bin_positions = np.digitize(self.raw_values, self.bin_edges) - 1
        bin_positions = np.clip(bin_positions, 0, BINS - 1)
        original_distribution = self.histogram_values[bin_positions]

        # The scaling ratio to ensure Mq >= p for all x
        scaling_factor = 1.02 * np.max(original_distribution / (fitted_distribution))

        # Initialise accepted array and remaining tracker
        accepted_vals = []
        remaining = num
        # print("Scaling factor found, time to populate t")
        # Loop until we have an array of size num
        while remaining > 0:
            # Random draws from the beta distribution
            sample: np.ndarray = beta.rvs(alpha, b, loc=0, scale=1, size=remaining)  # type: ignore

            # Uniform draws from 0 to 1
            uniform_samples = np.random.uniform(0, 1, size=remaining)

            # Construct the pdfs of the beta distribution and existing values
            sample_vals = beta.pdf(sample, alpha, b, loc=0, scale=1)
            indices = np.digitize(sample, self.bin_edges) - 1
            indices = np.clip(indices, 0, BINS - 1)
            original_vals = self.histogram_values[indices]

            # Mask for slicing valid values
            acceptance_mask = uniform_samples <= (
                original_vals / (scaling_factor * sample_vals)
            )
            accepted = sample[acceptance_mask]

            # If we get usable values, update corresponding arrays
            if len(accepted) > 0:
                # print("Populating some values")
                amount_needed = min(remaining, len(accepted))
                accepted_vals.append(accepted[:amount_needed])
                remaining -= amount_needed
        # print("Population done.")
        accepted_array = np.concatenate(accepted_vals)
        # bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        # f, b = np.histogram(fitted_distribution, BINS, T_RANGE, density=True)
        # print(scaling_factor)
        # plt.close()
        # vals, b = np.histogram(accepted_array, BINS, T_RANGE, density=True)
        # plt.plot(bin_centers, vals, label="Accepted values")
        # plt.plot(bin_centers, self.histogram_values, label="Original values")
        # plt.plot(
        #     bin_centers,
        #     scaling_factor * f,
        #     label="Scaled beta fit",
        # )
        # plt.plot(bin_centers, f, label="Base Beta fit")
        # plt.legend()
        # plt.savefig("tprime_plot.png", dpi=150)
        # sys.exit(0)
        return accepted_array

    def sample_z(self, num: int) -> np.ndarray:
        # Get our gaussian fit params
        mu, sigma = self.mean_and_std_from_hist()
        # print("Fitting done.")

        # Construct the pdf of z
        bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])
        fitted_distribution = norm.pdf(self.histogram_values, loc=mu, scale=sigma)
        bin_widths = np.diff(self.bin_edges)
        fitted_distribution /= np.sum(
            fitted_distribution * bin_widths
        )  # Renormalise it to sum to 1
        print(self.domain_min, self.domain_max)
        print(len(fitted_distribution), np.sum(fitted_distribution * bin_widths))
        print(np.sum(self.histogram_values * bin_widths))
        # Do the same thing as in the t distribution, get the indices first
        bin_positions = np.digitize(self.raw_values, self.bin_edges) - 1
        bin_positions = np.clip(bin_positions, 0, BINS - 1)
        # original_distribution = self.histogram_values[bin_positions]

        # The scaling ratio to ensure Mq >= p for all x
        scaling_mask = fitted_distribution >= 1e-8
        scaling_factor = 1.02 * np.max(
            self.histogram_values[scaling_mask] / fitted_distribution[scaling_mask]
        )
        print(scaling_factor)
        # Initialise accepted array and remaining tracker
        accepted_vals = []
        remaining = num
        # print("Scaling factor found, time to populate z")
        # Loop until we have an array of size num
        start = time()
        while remaining > 0:
            # Random draws from the gaussian distribution
            sample: np.ndarray = norm.rvs(loc=mu, scale=sigma, size=remaining)  # type: ignore

            # Uniform draws from 0 to 1
            uniform_samples = np.random.uniform(0, 1, size=remaining)

            # Construct the pdfs of the gaussian distribution and existing values
            sample_vals = norm.pdf(sample, loc=mu, scale=sigma)
            indices = np.digitize(sample, self.bin_edges) - 1
            indices = np.clip(indices, 0, BINS - 1)
            original_vals = self.histogram_values[indices]

            # Mask for slicing valid values
            acceptance_mask = uniform_samples <= (
                original_vals / (scaling_factor * sample_vals)
            )
            accepted = sample[acceptance_mask]

            # If we get usable values, update corresponding arrays
            if len(accepted) > 0:
                # print("Populating some values")
                amount_needed = min(remaining, len(accepted))
                accepted_vals.append(accepted[:amount_needed])
                remaining -= amount_needed
                # print(f"Accepted: {len(accepted)}, Remaining: {remaining}")
        end = time()
        print(f"Population done in {end - start:.3f}seconds.")
        accepted_array = np.concatenate(accepted_vals)
        accepted_dist = norm.pdf(accepted_array, loc=mu, scale=sigma)
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        f, b = np.histogram(fitted_distribution, BINS, Z_RANGE, density=True)
        print(f"z scaling factor: {scaling_factor}")
        plt.close()
        vals, b = np.histogram(accepted_array, BINS, Z_RANGE, density=True)
        plt.plot(bin_centers, vals, label="Accepted values")
        plt.plot(bin_centers, self.histogram_values, label="Original values")
        # plt.plot(
        #     bin_centers,
        #     2 * f,
        #     label="Scaled gaussian fit",
        # )
        # plt.plot(bin_centers, f, label="Base gaussian fit")
        plt.legend()
        plt.savefig("tprime_plot.png", dpi=150)
        sys.exit(0)
        return accepted_array


def center_z_distribution(Q_z: Probability_Distribution) -> Probability_Distribution:
    """Symmetrize and renormalize a binned Q(z) distribution in-place.

    The function enforces the physical symmetry Q(z) = Q(-z) by averaging the
    histogram values with their reversed order and renormalising the result so
    the histogram integrates to unity over the bin widths. The underlying
    `Probability_Distribution` object is updated and returned.

    Parameters
    ----------
    Q_z : Probability_Distribution
        Distribution object whose histogram will be symmetrised and updated.

    Returns
    -------
    Probability_Distribution
        The same `Q_z` instance after its histogram and CDF have been
        symmetrised and renormalised.

    Notes
    -----
    The function assumes `Q_z.histogram_values` and `Q_z.bin_edges` are valid
    and will call `Q_z.update` to replace the stored histogram with the
    symmetrised version.
    """
    bin_edges = Q_z.bin_edges
    dz = np.diff(bin_edges)

    hist_values = Q_z.histogram_values
    symmetrised_Qz = 0.5 * (hist_values + hist_values[::-1])
    symmetrised_Qz /= np.sum(symmetrised_Qz * dz)
    Q_z.update(symmetrised_Qz, bin_edges)
    # centered_z = Q_z - np.median(Q_z)
    # new_z = np.concatenate([centered_z, -centered_z])
    return Q_z


def extract_t_samples(P_t: Probability_Distribution, N: int) -> np.ndarray:
    """Generate a matrix of amplitude samples for the RG transformation.

    Draws 5 independent sets of N samples from the given P(t) distribution
    and arranges them into a matrix suitable for the RG transformation step.

    Parameters
    ----------
    P_t : Probability_Distribution
        Distribution object representing the current P(t) distribution.
    N : int
        Number of sample sets to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 5) containing the sampled amplitude values.
    """
    t1 = P_t.sample_t(N)
    t2 = P_t.sample_t(N)
    t3 = P_t.sample_t(N)
    t4 = P_t.sample_t(N)
    t5 = P_t.sample_t(N)

    t_sample = np.stack([t1, t2, t3, t4, t5], axis=1)
    return t_sample

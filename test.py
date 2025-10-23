import numpy as np
import matplotlib.pyplot as plt
from src.exponent_analysis import fit_z_peaks
from numpy.polynomial import polynomial


if __name__ == "__main__":
    x = np.array([0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3])
    # Shreyas 1e6
    # y = np.array(
    #     [
    #         0.039644874159733126,
    #         1.5505210730235135,
    #         2.184967056071726,
    #         2.4398749799713606,
    #         2.7071159482981573,
    #         2.870173599236527,
    #         3.324614240102175,
    #         3.430427172358917,
    #         3.3691117886847293,
    #     ]
    # ) / (2**7)
    # Jack 1e6
    # y = np.array(
    #     [
    #         0.22587169646620903,
    #         1.03502636494205,
    #         1.5059362351741359,
    #         1.6804564960941768,
    #         1.734522882413744,
    #         1.6953439669490638,
    #         1.7903342453996811,
    #         1.744985287777244,
    #         1.7953177955589765,
    #     ]
    # ) / (2**8)
    # Jack 1e7
    y = (
        np.array(
            [
                -0.34495857325697277,
                1.0951252526588873,
                1.6300433068491789,
                1.730254034172297,
                1.7749417406793284,
                1.6950390684666172,
                1.6551464100233773,
                1.7349894724218191,
                1.7099519185828302,
            ]
        )
        / 2**8
    )

    # result = linregress(x, y)
    ns, p = polynomial.Polynomial.fit(x, y, deg=1, full=True)
    resid = p[0]
    sst = float(np.dot(y, y))
    r2 = 1 - (resid / sst)
    coef = np.polyfit(x, y, 1)
    print(f"Polyfit coefficients: {coef}")
    func = np.poly1d(coef)
    plt.plot(x, func(x), label="polyfit")
    # slope = result.slope  # type: ignore
    # r2 = result.rvalue**2  # type: ignore
    s, sl, r = fit_z_peaks(x, y)
    lin_s, lin_r = fit_z_peaks(x, y, "linear")
    lin_nu = 8 * np.log(2) / np.log(np.abs(lin_s))

    print(
        f"Lin regress: absolute slope={np.abs(lin_s)} slope={lin_s} r2={lin_r} nu={lin_nu}"
    )

    nu = 8 * np.log(2) / np.log(s)
    print(f"Least squares: absolute slope={s} slope={sl} r2={r} nu={nu}")

    plt.scatter(x, y, marker="x", color="r", label="data points")
    plt.plot(x, lin_s * x, label="linregress")
    plt.plot(x, np.abs(lin_s) * x, label="abs linregress")
    plt.plot(x, s * x, label="ls absolute")
    plt.plot(x, sl * x, label="ls")
    plt.legend()
    plt.savefig("test5.png", dpi=300)

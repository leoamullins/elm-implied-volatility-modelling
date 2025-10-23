import time
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error

from elm.models.pricing.elm_pricer import OptionPricingELM
from elm.models.pricing.methods.cos import COSPricer
from elm.models.pricing.models.heston import HestonModel

# Default Heston model parameters
DEFAULT_HESTON_PARAMS: Dict[str, float] = {
    "S0": 100,
    "r": 0.02,  # Risk free rate
    "q": 0.00,  # dividend yield
    "v0": 0.04,  # Initial variance (20% volatility)
    "theta": 0.04,  # Long-term variance (20% volatility)
    "kappa": 2.0,  # Rate of mean reversion
    "sigma": 0.3,  # Volatility of variance
    "rho": -0.7,  # Correlation between asset and variance (typically negative)
}


def generate_iv_data(
    n_data_points: int = 100,
    def_heston_params: Dict[str, float] = DEFAULT_HESTON_PARAMS,
    K_range: Tuple[float, float] = (80.0, 120.0),
    T_range: Tuple[float, float] = (0.5, 3.0),
):
    start = time.time()
    K = np.linspace(K_range[0], K_range[1], num=n_data_points, endpoint=True)
    T = np.linspace(T_range[0], T_range[1], num=n_data_points, endpoint=True)

    K, T = np.meshgrid(K, T)
    iv_values = np.zeros_like(K)
    for i, j in np.ndindex(K.shape):
        ki = float(K[i, j])
        ti = float(T[i, j])
        hm = HestonModel(
            S0=def_heston_params["S0"],
            v0=def_heston_params["v0"],
            theta=def_heston_params["theta"],
            kappa=def_heston_params["kappa"],
            sigma=def_heston_params["sigma"],
            rho=def_heston_params["rho"],
            r=0.02,
            q=0,
        )

        cp = COSPricer()

        iv = cp.price_to_implied_vol(
            S0=def_heston_params["S0"],
            K=ki,
            T=ti,
            r=0.02,
            option_type="call",
            cf=hm.characteristic_function,
        )

        iv_values[i, j] = iv
    end = time.time()

    print(f"Completed analytical data forming in {end - start}s")
    return K, T, iv_values


def generate_elm_pred(
    heston_params: Dict[str, float] = DEFAULT_HESTON_PARAMS,
    num_data: int = 100,
    model=None,
):
    K, T, iv_values = generate_iv_data(n_data_points=num_data)

    X = np.zeros((K.ravel().shape[0], 10))
    shape = (X.shape[0],)
    start1 = time.time()
    print(
        f"Forming a training set with {X.shape[0]} data points and total shape {X.shape}."
    )

    K_values = K.ravel()
    T_values = T.ravel()

    X[:, 0] = np.full(shape, heston_params["S0"])
    X[:, 1] = K_values
    X[:, 2] = T_values
    X[:, 3] = np.full(shape, heston_params["r"])
    X[:, 4] = np.full(shape, heston_params["q"])
    X[:, 5] = np.full(shape, heston_params["v0"])
    X[:, 6] = np.full(shape, heston_params["theta"])
    X[:, 7] = np.full(shape, heston_params["kappa"])
    X[:, 8] = np.full(shape, heston_params["sigma"])
    X[:, 9] = np.full(shape, heston_params["rho"])

    y = iv_values.ravel()

    end1 = time.time()

    print(f"Training sets built in {end1 - start1}s.")

    if model is None:
        model = OptionPricingELM(
            n_hidden=4000,
            activation="tanh",
            scale=1,
            random_state=42,
            normalise_features=True,
            normalise_target=False,
            regularisation_param=1e-3,
            clip_negative=True,
            target_transform="none",
            forward_normalise=True,
            normalised_init=True,
        )

        print("Training model ...")
        start = time.time()
        model.fit(X, y)
        end = time.time()
        print(f"Trained in {end - start}s")
    else:
        model = model

    print("Model trained.")

    y_pred = model.predict(X)

    print(
        f"Predicted {y_pred.shape[0]} IV values.\nWith RMSE: {root_mean_squared_error(y, y_pred)}."
    )

    return y_pred, K, T


def plot_3d_surface(
    K: np.ndarray,
    T: np.ndarray,
    iv_surface: np.ndarray,
    title: str = "Implied Volatility Surface",
    save_path: Optional[str] = None,
):
    """
    Plot a simple 3D surface of implied volatility.

    Parameters
    ----------
    K : np.ndarray
        Strike price grid (2D meshgrid)
    T : np.ndarray
        Time to maturity grid (2D meshgrid)
    iv_surface : np.ndarray
        Implied volatility values (2D array or 1D array to reshape)
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    # If iv_surface is 1D, reshape it to match K and T
    if iv_surface.ndim == 1:
        iv_surface = iv_surface.reshape(K.shape)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        K, T, iv_surface, cmap="viridis", edgecolor="none", alpha=0.9, antialiased=True
    )

    # Labels and title
    ax.set_xlabel("Strike Price (K)", fontsize=12, labelpad=10)
    ax.set_ylabel("Time to Maturity (T)", fontsize=12, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Implied Volatility")

    # Set viewing angle
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Generate ELM predictions and plot
    model = joblib.load("models/elmtrainedsyntheticdata.pkl")
    start = time.time()
    y_pred, K, T = generate_elm_pred(num_data=100, model=model)
    total = time.time() - start

    print(f"\nTotal run time was {total:.2f}s.")

    # Plot the 3D surface
    print("\nPlotting 3D surface...")
    plot_3d_surface(
        K=K,
        T=T,
        iv_surface=y_pred,
        title="ELM Predicted Implied Volatility Surface",
        save_path="elm_iv_surface.pdf",
    )

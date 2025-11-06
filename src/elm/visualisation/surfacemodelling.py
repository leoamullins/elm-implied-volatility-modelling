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
    """
    Generate ELM predictions on a grid using a trained model.

    Parameters
    ----------
    heston_params : Dict[str, float]
        Heston model parameters to use for the grid
    num_data : int
        Number of points per dimension (creates num_data x num_data grid)
    model : OptionPricingELM
        Pre-trained ELM model

    Returns
    -------
    y_pred : np.ndarray
        Predicted IV values (flattened)
    K : np.ndarray
        Strike grid (2D meshgrid)
    T : np.ndarray
        Maturity grid (2D meshgrid)
    """
    # Generate grid
    K_range = (80.0, 120.0)
    T_range = (0.5, 3.0)
    K_1d = np.linspace(K_range[0], K_range[1], num=num_data, endpoint=True)
    T_1d = np.linspace(T_range[0], T_range[1], num=num_data, endpoint=True)
    K, T = np.meshgrid(K_1d, T_1d)

    # Create feature matrix
    X = np.zeros((K.ravel().shape[0], 10))
    shape = (X.shape[0],)

    K_values = K.ravel()
    T_values = T.ravel()

    X[:, 0] = np.full(shape, heston_params["S0"])
    X[:, 1] = K_values
    X[:, 2] = T_values
    X[:, 3] = np.full(shape, heston_params["r"])
    X[:, 4] = np.full(shape, heston_params["q"])
    X[:, 5] = np.full(shape, heston_params["v0"])
    X[:, 6] = np.full(shape, heston_params["kappa"])
    X[:, 7] = np.full(shape, heston_params["theta"])
    X[:, 8] = np.full(shape, heston_params["sigma"])
    X[:, 9] = np.full(shape, heston_params["rho"])

    # Predict using the trained model
    if model is None:
        raise ValueError("Model must be provided for prediction")

    y_pred = model.predict(X)

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
    from elm.models.pricing.elm_pricer import generate_heston_training_data

    print("=" * 70)
    print("ELM Implied Volatility Surface Generation")
    print("=" * 70)

    # 1. Generate analytical Heston training data using COS pricer
    print("\n1. Generating analytical Heston training data...")
    print("   Using COS pricer for exact Heston implied volatilities")
    start_data = time.time()

    X, y_iv = generate_heston_training_data(
        n_samples=10000,
        random_state=42,
        target_type="implied_volatility",
    )

    data_time = time.time() - start_data
    print(f"   Data generated in {data_time:.2f}s")
    print(f"   Training set: X.shape={X.shape}, y.shape={y_iv.shape}")
    print(f"   IV range: [{y_iv.min():.4f}, {y_iv.max():.4f}]")

    # 2. Train ELM model with optimal parameters

    try:
        model = joblib.load("models/elmtraininglargedataset.pkl")
        print("Model loaded from saved models.")
    except FileNotFoundError:
        print("Model file not found. Training new model:")
        model = OptionPricingELM(
            n_hidden=4000,
            activation="sine",
            scale=0.5,
            random_state=42,
            normalise_features=True,
            normalise_target=False,
            regularisation_param=1e-3,
            clip_negative=True,
            target_transform="none",
            forward_normalise=True,
            normalised_init=True,
        )

        start_train = time.time()
        model.fit(X, y_iv)
        train_time = time.time() - start_train
        print(f"   Model trained in {train_time:.2f}s")
    print("Model completed!")
    # 3. Evaluate on training data
    print("\n3. Evaluating model performance...")
    y_pred_train = model.predict(X[:1000])
    rmse = root_mean_squared_error(y_true=y_iv[:1000], y_pred=y_pred_train)
    print(f"   Training RMSE (1000 samples): {rmse:.6f}")

    # 4. Generate surface predictions
    print("\n4. Generating implied volatility surface...")
    start_surface = time.time()
    y_pred, K, T = generate_elm_pred(num_data=100, model=model)
    surface_time = time.time() - start_surface

    print(f"   Surface generated in {surface_time:.2f}s")
    print(f"   Surface shape: {y_pred.reshape(100, 100).shape}")
    print(f"   IV range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    # 5. Generate analytical surface for comparison
    print("\n5. Generating analytical Heston surface for comparison...")
    start_analytical = time.time()
    K_analytical, T_analytical, iv_analytical = generate_iv_data(n_data_points=100)
    analytical_time = time.time() - start_analytical
    print(f"   Analytical surface generated in {analytical_time:.2f}s")
    print(f"   IV range: [{iv_analytical.min():.4f}, {iv_analytical.max():.4f}]")

    # Compare surfaces
    y_pred_reshaped = y_pred.reshape(100, 100)
    surface_diff = y_pred_reshaped - iv_analytical
    mae_surface = np.mean(np.abs(surface_diff))
    rmse_surface = np.sqrt(np.mean(surface_diff**2))
    print(f"\n   Surface comparison metrics:")
    print(f"   - Mean Absolute Error: {mae_surface:.6f}")
    print(f"   - Root Mean Squared Error: {rmse_surface:.6f}")
    print(f"   - Max Absolute Difference: {np.max(np.abs(surface_diff)):.6f}")

    # 6. Plot both surfaces
    print("\n6. Plotting 3D surfaces...")

    # Create side-by-side comparison
    fig = plt.figure(figsize=(20, 6))

    # Plot 1: ELM surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(
        K, T, y_pred_reshaped, cmap="viridis", edgecolor="none", alpha=0.9, antialiased=True
    )
    ax1.set_xlabel("Strike Price (K)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Time to Maturity (T)", fontsize=10, labelpad=8)
    ax1.set_zlabel("Implied Volatility", fontsize=10, labelpad=8)
    ax1.set_title("ELM Predicted IV Surface", fontsize=12, pad=15)
    ax1.view_init(elev=20, azim=-60)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Plot 2: Analytical surface
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(
        K_analytical,
        T_analytical,
        iv_analytical,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )
    ax2.set_xlabel("Strike Price (K)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Time to Maturity (T)", fontsize=10, labelpad=8)
    ax2.set_zlabel("Implied Volatility", fontsize=10, labelpad=8)
    ax2.set_title("Analytical Heston IV Surface", fontsize=12, pad=15)
    ax2.view_init(elev=20, azim=-60)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Plot 3: Difference surface with improved scaling
    ax3 = fig.add_subplot(133, projection="3d")

    # Calculate symmetric limits for better visualization
    max_diff = np.max(np.abs(surface_diff))

    surf3 = ax3.plot_surface(
        K,
        T,
        surface_diff,
        cmap="RdBu_r",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
        vmin=-max_diff,
        vmax=max_diff,  # Symmetric color scale centered at zero
    )
    ax3.set_xlabel("Strike Price (K)", fontsize=10, labelpad=8)
    ax3.set_ylabel("Time to Maturity (T)", fontsize=10, labelpad=8)
    ax3.set_zlabel("IV Difference", fontsize=10, labelpad=8)
    ax3.set_title("Difference (ELM - Analytical)", fontsize=12, pad=15)
    ax3.view_init(elev=20, azim=-60)

    # Set z-axis limits to the actual difference range for better visibility
    ax3.set_zlim(-max_diff * 1.1, max_diff * 1.1)

    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig("iv_surface_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Also save individual surfaces
    plot_3d_surface(
        K=K,
        T=T,
        iv_surface=y_pred,
        title="ELM Predicted Implied Volatility Surface",
        save_path="docs/images/elm_iv_surface.png",
    )

    plot_3d_surface(
        K=K_analytical,
        T=T_analytical,
        iv_surface=iv_analytical,
        title="Analytical Heston Implied Volatility Surface",
        save_path="docs/images/analytical_iv_surface.png",
    )

    plot_3d_surface(
        K=K,
        T=T,
        iv_surface=surface_diff,
        title="Surface Difference (ELM - Analytical)",
        save_path="docs/images/difference_iv_surface.png",
    )

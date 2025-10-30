import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from elm.models.pricing.elm_pricer import OptionPricingELM

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def data_preprocessing(df):
    print(f"Original data shape: {df.shape}")

    df_clean = df.drop_duplicates()
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean["trade_iv"] > 0]
    df_clean = df_clean[df_clean["trade_iv"] < 3]
    df_clean = df_clean[~np.isinf(df_clean["trade_iv"])]

    print(f"After preprocessing: {df_clean.shape}")
    return df_clean


def create_feature_matrix(df):
    """
    Create the 10-feature matrix expected by the ELM.
    """
    print("Creating ELM-compatible feature matrix...")

    # Convert datetime columns
    df["quote_datetime"] = pd.to_datetime(df["quote_datetime"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Basic features
    S0 = (df["underlying_ask"] + df["underlying_bid"]) / 2
    K = df["strike"]
    days_to_expiry = (df["expiration"] - df["quote_datetime"]).dt.days
    T = days_to_expiry / 365.25

    # Risk-free rate
    risk_free_rate = 0.05
    r = np.full(len(df), risk_free_rate)

    # Dividend yield mapping
    div_map = {"SPY": 0.015, "SPX": 0.015, "QQQ": 0.006, "IWM": 0.012, "DIA": 0.018}
    q = df["underlying_symbol"].map(div_map).fillna(0.01)

    # Heston parameters (from synthetic data)
    synthetic_data = pd.read_csv("data/training_data_n100000_call_cos_X.csv")
    heston_params = {
        "v0": synthetic_data.iloc[:, 5].mean(),
        "theta": synthetic_data.iloc[:, 6].mean(),
        "kappa": synthetic_data.iloc[:, 7].mean(),
        "sigma": synthetic_data.iloc[:, 8].mean(),
        "rho": synthetic_data.iloc[:, 9].mean(),
    }

    v0 = np.full(len(df), heston_params["v0"])
    theta = np.full(len(df), heston_params["theta"])
    kappa = np.full(len(df), heston_params["kappa"])
    sigma = np.full(len(df), heston_params["sigma"])
    rho = np.full(len(df), heston_params["rho"])

    # Create feature matrix with ONLY the 10 expected features
    X = np.column_stack(
        [
            S0,  # Spot price
            K,  # Strike price
            T,  # Time to maturity
            r,  # Risk-free rate
            q,  # Dividend yield
            v0,  # Initial variance
            theta,  # Long-term variance
            kappa,  # Rate of mean reversion
            sigma,  # Volatility of variance
            rho,  # Correlation between asset and variance
        ]
    )

    print(f"Feature matrix shape: {X.shape}")
    print("Features: S0, K, T, r, q, v0, theta, kappa, sigma, rho")

    return X


def train_model(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = [
        # Small models (1K-3K)
        {
            "n_hidden": 1000,
            "activation": "tanh",
            "scale": 0.5,
            "regularisation_param": 1e-2,
        },
        {
            "n_hidden": 2000,
            "activation": "sine",
            "scale": 0.5,
            "regularisation_param": 1e-2,
        },
        {
            "n_hidden": 3000,
            "activation": "tanh",
            "scale": 1.0,
            "regularisation_param": 1e-3,
        },
        # Medium models (4K-6K)
        {
            "n_hidden": 4000,
            "activation": "sine",
            "scale": 1.0,
            "regularisation_param": 1e-3,
        },
        {
            "n_hidden": 5000,
            "activation": "tanh",
            "scale": 1.5,
            "regularisation_param": 1e-4,
        },
        {
            "n_hidden": 6000,
            "activation": "sine",
            "scale": 1.5,
            "regularisation_param": 1e-4,
        },
        # Large models (7K-10K)
        {
            "n_hidden": 7000,
            "activation": "tanh",
            "scale": 2.0,
            "regularisation_param": 1e-4,
        },
        {
            "n_hidden": 8000,
            "activation": "sine",
            "scale": 2.0,
            "regularisation_param": 1e-4,
        },
        {
            "n_hidden": 9000,
            "activation": "tanh",
            "scale": 2.5,
            "regularisation_param": 1e-5,
        },
        {
            "n_hidden": 10000,
            "activation": "sine",
            "scale": 2.5,
            "regularisation_param": 1e-5,
        },
        # Ultra-large models (12K-20K)
        {
            "n_hidden": 12000,
            "activation": "tanh",
            "scale": 3.0,
            "regularisation_param": 1e-5,
        },
        {
            "n_hidden": 15000,
            "activation": "sine",
            "scale": 3.0,
            "regularisation_param": 1e-5,
        },
        {
            "n_hidden": 18000,
            "activation": "tanh",
            "scale": 3.5,
            "regularisation_param": 1e-6,
        },
        {
            "n_hidden": 20000,
            "activation": "sine",
            "scale": 3.5,
            "regularisation_param": 1e-6,
        },
        # Maximum capacity
        {
            "n_hidden": 25000,
            "activation": "tanh",
            "scale": 4.0,
            "regularisation_param": 1e-6,
        },
    ]

    trained_models = []
    all_predictions = []

    print("\n Ensemble training... \n")

    for model in models:
        print(f"Training model with {model['n_hidden']} hidden neurons...")
        model = OptionPricingELM(
            n_hidden=model["n_hidden"],
            activation=model["activation"],
            scale=model["scale"],
            regularisation_param=model["regularisation_param"],
            random_state=random_state,
            normalise_features=True,
            normalise_target=False,
            forward_normalise=True,
            normalised_init=True,
        )
        model.fit(X_train, y_train)
        trained_models.append(model)
        pred = model.predict(X_test)
        all_predictions.append(pred)

    # Ensemble prediction
    predictions_array = np.array(all_predictions)
    ensemble_pred = np.mean(predictions_array, axis=0)
    uncertainty = np.std(predictions_array, axis=0)

    return (
        trained_models,
        all_predictions,
        X_test,
        y_test,
        ensemble_pred,
        uncertainty,
    )


def evaluate_model(y_true, y_pred, uncertainty=None):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    max_error = np.max(np.abs(y_true - y_pred))

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Max Error: {max_error:.6f}")

    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"Correlation: {correlation:.6f}")

    # Residual analysis
    residuals = y_true - y_pred
    print("\nResidual Analysis:")
    print(f"Residual mean: {residuals.mean():.6f}")
    print(f"Residual std: {residuals.std():.6f}")
    print(f"Residual range: {residuals.min():.6f} to {residuals.max():.6f}")

    # Uncertainty analysis
    if uncertainty is not None:
        print("\nUncertainty Analysis:")
        print(f"Mean uncertainty: {uncertainty.mean():.6f}")
        print(f"Uncertainty std: {uncertainty.std():.6f}")
        print(f"Uncertainty range: {uncertainty.min():.6f} to {uncertainty.max():.6f}")

        # Coverage analysis
        coverage_68 = np.mean(np.abs(residuals) <= uncertainty)
        coverage_95 = np.mean(np.abs(residuals) <= 2 * uncertainty)
        print(f"68% coverage: {coverage_68:.2%}")
        print(f"95% coverage: {coverage_95:.2%}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "max_error": max_error,
        "correlation": correlation,
    }


def plot_results(y_true, y_pred, uncertainty=None, save_path=None):
    """
    Create comprehensive plots for model evaluation results.

    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Uncertainty estimates (optional)
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("ELM Option Pricing Model Results", fontsize=16, fontweight="bold")

    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0, 0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")
    axes[0, 0].set_title("Actual vs Predicted")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals vs Predicted")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residuals histogram
    axes[0, 2].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 2].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[0, 2].set_xlabel("Residuals")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Residuals Distribution")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Time series comparison (first 200 points)
    n_points = min(200, len(y_true))
    x_axis = range(n_points)
    axes[1, 0].plot(x_axis, y_true[:n_points], label="Actual", alpha=0.8, linewidth=1)
    axes[1, 0].plot(
        x_axis, y_pred[:n_points], label="Predicted", alpha=0.8, linewidth=1
    )
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("Values")
    axes[1, 0].set_title("Time Series Comparison (First 200 Points)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Q-Q plot for residuals
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot of Residuals")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Uncertainty plot (if available)
    if uncertainty is not None:
        # Plot uncertainty bands
        sorted_indices = np.argsort(y_pred)
        sorted_pred = y_pred[sorted_indices]
        sorted_uncertainty = uncertainty[sorted_indices]

        axes[1, 2].fill_between(
            sorted_pred,
            sorted_pred - sorted_uncertainty,
            sorted_pred + sorted_uncertainty,
            alpha=0.3,
            label="±1σ Uncertainty",
        )
        axes[1, 2].fill_between(
            sorted_pred,
            sorted_pred - 2 * sorted_uncertainty,
            sorted_pred + 2 * sorted_uncertainty,
            alpha=0.2,
            label="±2σ Uncertainty",
        )
        axes[1, 2].scatter(y_pred, y_true, alpha=0.6, s=10, label="Data Points")
        axes[1, 2].set_xlabel("Predicted Values")
        axes[1, 2].set_ylabel("Actual Values")
        axes[1, 2].set_title("Prediction with Uncertainty Bands")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Error distribution
        axes[1, 2].hist(np.abs(residuals), bins=50, alpha=0.7, edgecolor="black")
        axes[1, 2].set_xlabel("Absolute Error")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].set_title("Absolute Error Distribution")
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("PLOTTING SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(y_true)}")
    print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred) ** 2)):.6f}")
    print(f"MAE: {np.mean(np.abs(y_true - y_pred)):.6f}")
    print(f"R²: {r2_score(y_true, y_pred):.6f}")
    print(f"Correlation: {np.corrcoef(y_true, y_pred)[0, 1]:.6f}")
    if uncertainty is not None:
        print(f"Mean uncertainty: {uncertainty.mean():.6f}")
        print(f"Uncertainty std: {uncertainty.std():.6f}")


if __name__ == "__main__":
    df = pd.read_csv("data/options_data_20k.csv")
    df_clean = data_preprocessing(df)

    # Create proper feature matrix with only numeric features
    X = create_feature_matrix(df_clean)
    y = df_clean["trade_iv"]
    trained_models, predictions, X_test, y_test, ensemble_pred, uncertainty = (
        train_model(X, y)
    )
    evaluate_model(y_test, ensemble_pred, uncertainty)

    # Create comprehensive plots
    plot_results(
        y_test, ensemble_pred, uncertainty, save_path="elm_option_pricing_results.png"
    )

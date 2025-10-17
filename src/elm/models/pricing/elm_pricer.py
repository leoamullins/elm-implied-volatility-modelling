from __future__ import annotations

import time
import numpy as np
from typing import Dict, Tuple, Optional, Literal

# prefer sklearn standard scaler
try:
    from sklearn.preprocessing import StandardScaler as _SkStandardScaler
except Exception:

    class _SkStandardScaler:
        def __init__(self) -> None:
            self.mean_ = None
            self.scale_ = None

        def fit(self, X: np.ndarray) -> "_SkStandardScaler":
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.mean_ = np.mean(X, axis=0)
            std = np.std(X, axis=0)

            # avoid division by zero
            std = np.where(std == 0, 1.0, std)
            self.scale_ = std
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise ValueError(
                    "This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
                )
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return self.fit(X).transform(X)

        def inverse_transform(self, X: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise ValueError(
                    "This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
                )
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X * self.scale_ + self.mean_


from elm.models.core.elm_base import ELM
from elm.models.pricing.models.heston import HestonModel
from elm.models.pricing.methods.cos import COSPricer

FEATURE_NAMES = [
    "S0",  # Spot price
    "K",  # Strike price
    "T",  # Time to maturity
    "r",  # Risk-free rate
    "q",  # Dividend yield
    "v0",  # Initial variance
    "theta",  # Long-term variance
    "kappa",  # Rate of mean reversion
    "sigma",  # Volatility of variance
    "rho",  # Correlation between asset and variance
]


class OptionPricingELM:
    def __init__(
        self,
        n_hidden: int = 200,
        activation: str = "tanh",
        random_state: Optional[int] = None,
        scale: float = 1.0,
        normalised_init: bool = False,
        regularisation: bool = True,
        regularisation_param: float = 1e-3,
        normalise_features: bool = True,
        normalise_target: bool = False,
        clip_negative: bool = True,
        target_transform: Literal["none", "by_strike", "by_spot", "by_forward"] = None,
        forward_normalise: bool = True,
    ) -> None:
        self.n_hidden = int(n_hidden)
        self.activation = str(activation)
        self.random_state = random_state
        self.scale = float(scale)
        self.normalised_init = bool(normalised_init)
        self.regularisation = bool(regularisation)
        self.regularisation_param = float(regularisation_param)
        self.normalise_features = bool(normalise_features)
        self.normalise_target = bool(normalise_target)
        self.clip_negative = bool(clip_negative)
        self.target_transform = str(target_transform)
        self.forward_normalise = bool(forward_normalise)

        # CORE ELM
        self.elm = ELM(
            n_hidden=self.n_hidden,
            activation=self.activation,
            random_state=self.random_state,
            scale=self.scale,
            normalised_init=self.normalised_init,
        )

        # PREPROCESSING
        self.feature_scaler = _SkStandardScaler() if self.normalise_features else None
        self.target_scaler = _SkStandardScaler() if self.normalise_target else None

        # state
        self.is_fitted = False
        self.option_type = None
        self.feature_names = FEATURE_NAMES
        self._last_training_time = None
        self._fitted_feature_dim = None
        self._target_transform_mode = self.target_transform

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        expected_features = 7 if self.forward_normalise else len(self.feature_names)
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")

        # sanity checks
        if np.any(X[:, 0] <= 0):
            raise ValueError("Spot price S0 must be positive.")
        if np.any(X[:, 1] <= 0):
            raise ValueError("Strike price K must be positive.")
        if np.any(X[:, 2] <= 0):
            raise ValueError("Time to maturity T must be positive.")

        return X

    def _apply_target_transform_for_fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        mode = (self.target_transform or "none").lower()
        self._target_transform_mode = mode

        if mode == "none":
            return y

        S0 = X[:, 0]
        K = X[:, 1]
        T = X[:, 2]
        r = X[:, 3]
        q = X[:, 4]

        if mode == "by_strike":
            denom = K
        elif mode == "by_spot":
            denom = S0
        elif mode == "by_forward":
            denom = S0 * np.exp((r - q) * T)
        else:
            raise ValueError(
                "target_transform must be one of: 'none', 'by_strike', 'by_spot', 'by_forward'"
            )

        denom = np.asarray(denom).reshape(-1, 1)
        return y / denom

    def _invert_target_transform(self, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mode = (self._target_transform_mode or "none").lower()

        if mode == "none":
            return y_pred

        S0 = X[:, 0]
        K = X[:, 1]
        T = X[:, 2]
        r = X[:, 3]
        q = X[:, 4]

        if mode == "by_strike":
            denom = K
        elif mode == "by_spot":
            denom = S0
        elif mode == "by_forward":
            denom = S0 * np.exp((r - q) * T)
        else:
            raise ValueError(
                "target_transform must be one of: 'none', 'by_strike', 'by_spot', 'by_forward'"
            )

        denom = np.asarray(denom).reshape(-1, 1)
        return np.asarray(y_pred).reshape(-1, 1) * denom

    def _apply_forward_normalisation(self, X: np.ndarray) -> np.ndarray:
        """
        Apply forward normalization and keep only relevant inputs:
        (k = K/F, T, v0, kappa, theta, sigma, rho)
        """
        X = np.asarray(X, dtype=float).copy()

        S0 = X[:, 0]
        K = X[:, 1]
        T = X[:, 2]
        r = X[:, 3]
        q = X[:, 4]
        v0 = X[:, 5]
        theta = X[:, 6]
        kappa = X[:, 7]
        sigma = X[:, 8]
        rho = X[:, 9]

        F = S0 * np.exp((r - q) * T)
        k = K / F  # normalized strike

        # Return only the subset of features used in the paper
        return np.column_stack([k, T, v0, kappa, theta, sigma, rho])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        option_type: Literal["call", "put"] = "call",
    ) -> OptionPricingELM:
        # Always work with raw inputs for y-transform
        X_raw = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        self.option_type = option_type

        # Feature transform
        if self.forward_normalise:
            X_model = self._apply_forward_normalisation(X_raw)
            assert X_model.shape[1] == 7
        else:
            X_model = self._validate_features(X_raw)

        # Scale inputs if enabled
        if self.normalise_features and self.feature_scaler:
            X_scaled = self.feature_scaler.fit_transform(X_model)
        else:
            X_scaled = X_model

        # Target transform uses *raw* X (not the reduced one!)
        y_pre = self._apply_target_transform_for_fit(X_raw, y)
        if self.normalise_target and self.target_scaler:
            y_scaled = self.target_scaler.fit_transform(y_pre)
        else:
            y_scaled = y_pre

        # Train ELM
        self.elm = ELM(
            n_hidden=self.n_hidden,
            activation=self.activation,
            random_state=self.random_state,
            scale=self.scale,
            normalised_init=self.normalised_init,
        )
        start = time.perf_counter()
        self.elm.fit(
            X_scaled,
            y_scaled,
            regularisation=self.regularisation,
            regularisation_param=self.regularisation_param,
        )
        self._last_training_time = time.perf_counter() - start
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before making predictions.")

        X_raw = np.asarray(X, dtype=float)
        X_model = (
            self._apply_forward_normalisation(X_raw)
            if self.forward_normalise
            else X_raw
        )

        if self.normalise_features and self.feature_scaler:
            X_scaled = self.feature_scaler.transform(X_model)
        else:
            X_scaled = X_model

        y_pred_scaled = self.elm.predict(X_scaled)

        # Undo target scaling using original (10-column) X
        if self.normalise_target and self.target_scaler:
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        else:
            y_pred = y_pred_scaled.reshape(-1, 1)

        # Undo target transform (reapply F0, S0, etc.)
        y_pred = self._invert_target_transform(X_raw, y_pred)
        return np.maximum(y_pred.ravel(), 0.0) if self.clip_negative else y_pred.ravel()

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        metrics: Tuple[str, ...] = ("rmse", "mae", "mape"),
    ) -> Dict[str, float]:
        y_pred = self.predict(X)
        y_true = np.asarray(y_true, dtype=float).flatten()

        results = {}

        if "rmse" in metrics:
            results["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        if "mae" in metrics:
            results["mae"] = float(np.mean(np.abs(y_true - y_pred)))

        if "mape" in metrics:
            mask = y_true != 0
            results["mape"] = (
                float(
                    np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                )
                if np.any(mask)
                else float("nan")
            )

        if "r2" in metrics:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            results["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

        if "max_error" in metrics:
            results["max_error"] = float(np.max(np.abs(y_true - y_pred)))

        return results

    def compare_with_analytical(
        self,
        X: np.ndarray,
        method: Literal["cos", "monte_carlo", "fourier"] = "cos",
        comparison_mode: Literal["price", "implied_volatility"] = "price",
        N: int = 256,
        L: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_fitted is False:
            raise ValueError("The model must be fitted before comparison.")

        # Validate raw input features (should always be 10 for Heston model)
        X_raw = np.asarray(X, dtype=float)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        if X_raw.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X_raw.shape[1]}"
            )

        # Basic sanity checks on raw features
        if np.any(X_raw[:, 0] <= 0):
            raise ValueError("Spot price S0 must be positive.")
        if np.any(X_raw[:, 1] <= 0):
            raise ValueError("Strike price K must be positive.")
        if np.any(X_raw[:, 2] <= 0):
            raise ValueError("Time to maturity T must be positive.")

        # Get ELM predictions (always prices first, then convert if needed)
        elm_prices = self.predict(X_raw)

        analytical_prices = np.zeros(len(X_raw), dtype=float)

        use_cos = method.lower() == "cos"
        cos_pricer = COSPricer(N=N, L=L) if use_cos else None

        # Row by row pricing
        for i, row in enumerate(X_raw):
            S0, K, T, r, q, v0, kappa, theta, sigma, rho = [float(x) for x in row]

            heston_model = HestonModel(
                S0=S0,
                r=r,
                q=q,
                v0=v0,
                theta=theta,
                kappa=kappa,
                sigma=sigma,
                rho=rho,
            )

            if use_cos:
                cf = heston_model.characteristic_function
                analytical_prices[i] = cos_pricer.price(
                    cf=cf,
                    S0=S0,
                    K=K,
                    T=T,
                    r=r,
                    option_type=self.option_type or "call",
                )
            elif method.lower() == "monte_carlo":
                raise NotImplementedError("Monte Carlo pricing not implemented yet.")
            else:
                from elm.models.pricing.methods.fourier import FourierPricer

                fp = FourierPricer(heston_model)
                analytical_prices[i] = float(
                    fp.price(
                        K=K,
                        T=T,
                        r=r,
                        option_type=self.option_type or "call",
                    )
                )

        # Convert to implied volatilities if requested
        if comparison_mode.lower() == "implied_volatility":
            from elm.models.pricing.methods.black_scholes import BlackScholes

            # Convert ELM prices to implied volatilities
            elm_ivs = np.zeros(len(X_raw), dtype=float)
            analytical_ivs = np.zeros(len(X_raw), dtype=float)

            for i, row in enumerate(X_raw):
                S0, K, T, r, q = row[:5]
                option_type = self.option_type or "call"

                # Convert ELM price to IV
                try:
                    elm_ivs[i] = BlackScholes.implied_volatility(
                        elm_prices[i], S0, K, T, r, option_type, q
                    )
                except Exception:
                    elm_ivs[i] = np.nan

                # Convert analytical price to IV
                try:
                    analytical_ivs[i] = BlackScholes.implied_volatility(
                        analytical_prices[i], S0, K, T, r, option_type, q
                    )
                except Exception:
                    analytical_ivs[i] = np.nan

            return elm_ivs, analytical_ivs
        else:
            # Return prices as before
            return elm_prices, analytical_prices

    def predict_implied_volatility(self, X: np.ndarray) -> np.ndarray:
        """
        Predict implied volatilities directly by converting ELM price predictions to IV.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction

        Returns
        -------
        np.ndarray
            Predicted implied volatilities
        """
        from elm.models.pricing.methods.black_scholes import BlackScholes

        X_raw = np.asarray(X, dtype=float)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        if X_raw.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X_raw.shape[1]}"
            )

        # Get ELM price predictions
        elm_prices = self.predict(X_raw)

        # Convert prices to implied volatilities
        ivs = np.zeros(len(X_raw), dtype=float)

        for i, row in enumerate(X_raw):
            S0, K, T, r, q = row[:5]
            option_type = self.option_type or "call"

            try:
                ivs[i] = BlackScholes.implied_volatility(
                    elm_prices[i], S0, K, T, r, option_type, q
                )
            except:
                ivs[i] = np.nan

        return ivs

    def price_from_implied_volatility(self, X: np.ndarray, iv: float) -> float:
        """
        Convert implied volatility back to option price using Black-Scholes.

        Parameters
        ----------
        X : array-like, shape (n_features,)
            Input features (S0, K, T, r, q, ...)
        iv : float
            Implied volatility

        Returns
        -------
        float
            Option price
        """
        from elm.models.pricing.methods.black_scholes import BlackScholes

        X_raw = np.asarray(X, dtype=float)
        if X_raw.ndim == 1 and len(X_raw) == 1:
            X_raw = X_raw.reshape(1, -1)
        elif X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        S0, K, T, r, q = X_raw[0, :5]
        option_type = self.option_type or "call"

        return BlackScholes.price(S0, K, T, r, iv, option_type, q)

    def get_training_time(self) -> float:
        if self._last_training_time is None:
            raise ValueError("The model has not been trained yet.")
        return float(self._last_training_time)

    def save_model(self, filepath: str) -> None:
        import joblib

        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath: str) -> OptionPricingELM:
        with open(filepath, "rb") as f:
            import joblib

            return joblib.load(f)

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError(
                "The model must be fitted before computing feature importance."
            )

        X = self._validate_features(X)
        y = np.asarray(y, dtype=float).flatten()

        baseline_rmse = self.evaluate(X, y, metrics=("rmse",))["rmse"]

        importances = {}
        rng = np.random.default_rng()

        for feature_idx, feature_name in enumerate(self.feature_names):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, feature_idx])
            permuted_rmse = self.evaluate(X_permuted, y, metrics=("rmse",))["rmse"]
            importances[feature_name] = float(permuted_rmse - baseline_rmse)

        return importances

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"OptionPricingELM(n_hidden={self.n_hidden}, activation='{self.activation}', "
            f"random_state={self.random_state}, scale={self.scale}, "
            f"regularisation={self.regularisation}, regularisation_param={self.regularisation_param}, "
            f"normalise_features={self.normalise_features}, normalise_target={self.normalise_target}, "
            f"status={fitted_str})"
        )


# ============================================================================
# Data Generation Utilities
# ============================================================================


def generate_heston_training_data(
    n_samples: int = 10000,
    option_type: Literal["call", "put"] = "call",
    pricing_method: Literal["cos", "fourier", "monte_carlo"] = "cos",
    target_type: Literal["price", "implied_volatility"] = "implied_volatility",
    random_state: Optional[int] = None,
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for option pricing using Heston model.
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    option_type : str
        'call' or 'put'
    pricing_method : str
        Method to compute ground truth: 'cos', 'fourier', 'monte_carlo'
    target_type : str
        'price' for option prices or 'implied_volatility' for implied volatilities
    random_state : int, optional
        Random seed
    parameter_ranges : dict, optional
        Custom ranges for parameters. Default ranges are used if None.
    Returns
    -------
    X : array, shape (n_samples, 10)
        Feature matrix
    y : array, shape (n_samples,)
        Target option prices or implied volatilities
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Default parameter ranges (realistic for equity options)
    default_ranges: Dict[str, Tuple[float, float]] = {
        "S0": (80.0, 120.0),  # Spot price around 100
        "K": (80.0, 120.0),  # Strike price
        "T": (0.1, 3.0),  # Maturity: 1 month to 2 years
        "r": (0.0, 0.05),  # Risk-free rate: 0% to 5%
        "q": (0.0, 0.00),  # Dividend yield: 0%
        "v0": (0.01, 0.45),  # Initial variance: 10% to 30% vol
        "kappa": (0.1, 4.0),  # Mean reversion speed
        "theta": (0.01, 0.09),  # Long-run variance
        "sigma": (0.1, 0.4),  # Vol of vol
        "rho": (-1.0, -0.1),  # Correlation (typically negative)
    }

    if parameter_ranges is not None:
        # Update only provided keys while keeping canonical order
        for key, rng in parameter_ranges.items():
            if key not in default_ranges:
                raise ValueError(f"Unknown parameter in parameter_ranges: {key}")
            low, high = float(rng[0]), float(rng[1])
            if low >= high:
                raise ValueError(
                    f"parameter_ranges['{key}'] must satisfy low < high, got {rng}"
                )
            default_ranges[key] = (low, high)

    # Generate random parameters in deterministic column order
    X = np.zeros((n_samples, len(FEATURE_NAMES)))
    for i, param in enumerate(FEATURE_NAMES):
        low, high = default_ranges[param]
        X[:, i] = np.random.uniform(low, high, n_samples)

    # Generate ground truth prices
    y = np.zeros(n_samples)

    if pricing_method == "cos":
        pricer = COSPricer(N=256, L=10.0)

    for i in range(n_samples):
        S0, K, T, r, q, v0, kappa, theta, sigma, rho = X[i]

        # Create Heston model
        heston = HestonModel(
            S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho, r=r, q=q
        )

        # Price option
        if pricing_method == "cos":
            cf = heston.characteristic_function
            option_price = pricer.price(K=K, T=T, r=r, cf=cf, option_type=option_type)

            if target_type == "implied_volatility":
                # Convert price to implied volatility
                y[i] = pricer.implied_volatility(
                    market_price=option_price,
                    S0=S0,
                    K=K,
                    T=T,
                    r=r,
                    option_type=option_type,
                    q=q,
                )
            else:
                y[i] = option_price

        elif pricing_method == "fourier":
            from elm.models.pricing.methods.fourier import FourierPricer

            fp = FourierPricer(heston)
            option_price = float(fp.price(K=K, T=T, option_type=option_type))

            if target_type == "implied_volatility":
                # Convert price to implied volatility using Black-Scholes
                from elm.models.pricing.methods.black_scholes import BlackScholes

                y[i] = BlackScholes.implied_volatility(
                    option_price, S0, K, T, r, option_type, q
                )
            else:
                y[i] = option_price

        elif pricing_method == "monte_carlo":
            from elm.models.pricing.methods.monte_carlo import MonteCarlo

            mc = MonteCarlo(heston)
            result = mc.price(K=K, T=T, option_type=option_type, n_paths=50000, seed=i)
            option_price = float(result["price"])

            if target_type == "implied_volatility":
                # Convert price to implied volatility using Black-Scholes
                from elm.models.pricing.methods.black_scholes import BlackScholes

                y[i] = BlackScholes.implied_volatility(
                    option_price, S0, K, T, r, option_type, q
                )
            else:
                y[i] = option_price

        else:
            raise ValueError(f"Unknown pricing method: {pricing_method}")

        if (i) % 1000 == 0:
            print(f"Completed {i} / {n_samples}!")

    return X, y


def create_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets.
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Splits must sum to 1.0")

    if random_state is not None:
        np.random.seed(random_state)

    n = len(X)
    indices = np.random.permutation(n)

    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        np.asarray(y)[train_idx],
        np.asarray(y)[val_idx],
        np.asarray(y)[test_idx],
    )

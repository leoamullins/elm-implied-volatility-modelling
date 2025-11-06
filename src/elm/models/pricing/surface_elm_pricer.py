"""
Enhanced ELM pricer specifically designed for IV surface modelling.
Includes advanced regularisation and surface-aware features.
"""

import numpy as np
from typing import Optional, Literal, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from elm.models.core.elm_base import ELM
from elm.models.pricing.elm_pricer import OptionPricingELM


class SurfaceELMPricer(OptionPricingELM):
    """
    Enhanced ELM pricer optimised for IV surface modelling.

    Key improvements:
    1. Surface-aware feature engineering
    2. Advanced regularisation techniques
    3. Multi-scale architecture
    4. Surface smoothness constraints
    """

    def __init__(
        self,
        n_hidden: int = 2000,
        activation: str = "sine",
        random_state: Optional[int] = None,
        scale: float = 0.5,
        normalised_init: bool = True,
        regularisation: bool = True,
        regularisation_param: float = 1e-3,
        normalise_features: bool = True,
        normalise_target: bool = False,
        clip_negative: bool = True,
        target_transform: Literal["none", "by_strike", "by_spot", "by_forward"] = None,
        forward_normalise: bool = True,
        # Surface-specific parameters
        surface_smoothness: float = 1e-2,
        multi_scale: bool = True,
        n_scales: int = 3,
        surface_regularisation: bool = True,
    ):
        super().__init__(
            n_hidden=n_hidden,
            activation=activation,
            random_state=random_state,
            scale=scale,
            normalised_init=normalised_init,
            regularisation=regularisation,
            regularisation_param=regularisation_param,
            normalise_features=normalise_features,
            normalise_target=normalise_target,
            clip_negative=clip_negative,
            target_transform=target_transform,
            forward_normalise=forward_normalise,
        )

        # Surface-specific parameters
        self.surface_smoothness = surface_smoothness
        self.multi_scale = multi_scale
        self.n_scales = n_scales
        self.surface_regularisation = surface_regularisation

        # Surface-specific attributes
        self.surface_scalers = {}
        self.scale_weights = None

    def _create_surface_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create surface-aware features that capture cross-dimensional dependencies.

        Parameters:
        -----------
        X : array, shape (n_samples, 10)
            Input features [S0, K, T, r, q, v0, theta, kappa, sigma, rho]

        Returns:
        --------
        X_enhanced : array, shape (n_samples, n_enhanced_features)
            Enhanced features including surface-aware transformations
        """
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

        # Basic features
        features = [X]

        # Moneyness features
        moneyness = K / S0
        log_moneyness = np.log(moneyness)
        features.append(moneyness.reshape(-1, 1))
        features.append(log_moneyness.reshape(-1, 1))

        # Time features
        sqrt_time = np.sqrt(T)
        log_time = np.log(T + 1e-6)  # Avoid log(0)
        features.append(sqrt_time.reshape(-1, 1))
        features.append(log_time.reshape(-1, 1))

        # Volatility features
        vol_ratio = v0 / theta
        vol_speed = kappa * theta
        features.append(vol_ratio.reshape(-1, 1))
        features.append(vol_speed.reshape(-1, 1))

        # Cross-dimensional features (key for surface modelling)
        # Strike-time interactions
        strike_time_interaction = log_moneyness * sqrt_time
        features.append(strike_time_interaction.reshape(-1, 1))

        # Volatility-strike interactions
        vol_strike_interaction = v0 * log_moneyness
        features.append(vol_strike_interaction.reshape(-1, 1))

        # Rate-time interactions
        rate_time_interaction = r * T
        features.append(rate_time_interaction.reshape(-1, 1))

        # Surface curvature features
        strike_curvature = log_moneyness**2
        time_curvature = T**2
        features.append(strike_curvature.reshape(-1, 1))
        features.append(time_curvature.reshape(-1, 1))

        # Combine all features
        X_enhanced = np.hstack(features)

        return X_enhanced

    def _create_multi_scale_features(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multi-scale features for better surface representation.

        Returns:
        --------
        X_scales : list of arrays
            Features at different scales
        scale_weights : array
            Weights for combining scales
        """
        X_scales = []

        # Original scale
        X_scales.append(X)

        # Coarse scale (smoothed)
        if self.n_scales > 1:
            X_coarse = self._smooth_features(X, factor=2)
            X_scales.append(X_coarse)

        # Fine scale (enhanced detail)
        if self.n_scales > 2:
            X_fine = self._enhance_features(X)
            X_scales.append(X_fine)

        # Initialize scale weights
        scale_weights = np.ones(len(X_scales)) / len(X_scales)

        return X_scales, scale_weights

    def _smooth_features(self, X: np.ndarray, factor: float = 2.0) -> np.ndarray:
        """Apply smoothing to features for coarse scale."""
        # Simple moving average smoothing
        from scipy.ndimage import gaussian_filter1d

        X_smooth = X.copy()
        for i in range(X.shape[1]):
            X_smooth[:, i] = gaussian_filter1d(X[:, i], sigma=factor)

        return X_smooth

    def _enhance_features(self, X: np.ndarray) -> np.ndarray:
        """Enhance features for fine scale."""
        # Add noise for robustness
        noise_scale = 0.01 * np.std(X, axis=0)
        noise = np.random.normal(0, noise_scale, X.shape)
        X_enhanced = X + noise

        return X_enhanced

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurfaceELMPricer":
        """
        Enhanced fit method with surface-aware training.
        """
        # Create enhanced features
        X_enhanced = self._create_surface_features(X)

        if self.multi_scale:
            # Multi-scale training
            X_scales, self.scale_weights = self._create_multi_scale_features(X_enhanced)

            # Train on each scale
            self._train_multi_scale(X_scales, y)
        else:
            # Standard training with enhanced features
            super().fit(X_enhanced, y)

        return self

    def _train_multi_scale(self, X_scales: list, y: np.ndarray) -> None:
        """Train multi-scale model."""
        # For simplicity, use the finest scale for now
        # In practice, you'd combine predictions from all scales
        X_final = X_scales[-1]  # Use the most detailed scale

        super().fit(X_final, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Enhanced prediction with surface smoothness.
        """
        # Create enhanced features
        X_enhanced = self._create_surface_features(X)

        # Make predictions
        y_pred = super().predict(X_enhanced)

        # Apply surface smoothness if requested
        if self.surface_regularisation:
            y_pred = self._apply_surface_smoothness(X, y_pred)

        return y_pred

    def _apply_surface_smoothness(
        self, X: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Apply surface smoothness constraints to predictions.
        """
        # Group predictions by surface (assuming X contains surface identifiers)
        # For now, apply simple smoothing
        from scipy.ndimage import gaussian_filter1d

        y_smooth = gaussian_filter1d(y_pred, sigma=self.surface_smoothness)

        return y_smooth

    def fit_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
        iv_surface: np.ndarray,
    ) -> "SurfaceELMPricer":
        """
        Fit model to a complete IV surface.

        Parameters:
        -----------
        strikes : array
            Strike prices
        maturities : array
            Time to maturity
        market_params : dict
            Market parameters (S0, r, q, v0, theta, kappa, sigma, rho)
        iv_surface : array, shape (n_maturities, n_strikes)
            Implied volatility surface
        """
        # Create training data from surface
        X_surface, y_surface = self._surface_to_training_data(
            strikes, maturities, market_params, iv_surface
        )

        # Fit with surface-specific regularization
        return self.fit(X_surface, y_surface)

    def _surface_to_training_data(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
        iv_surface: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert surface data to training format."""
        features = []
        targets = []

        S0 = market_params["S0"]
        r = market_params["r"]
        q = market_params["q"]
        v0 = market_params["v0"]
        theta = market_params["theta"]
        kappa = market_params["kappa"]
        sigma = market_params["sigma"]
        rho = market_params["rho"]

        for i, maturity in enumerate(maturities):
            for j, strike in enumerate(strikes):
                feature = [S0, strike, maturity, r, q, v0, theta, kappa, sigma, rho]
                features.append(feature)
                targets.append(iv_surface[i, j])

        return np.array(features), np.array(targets)

    def predict_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
    ) -> np.ndarray:
        """
        Predict complete IV surface.

        Returns:
        --------
        iv_surface : array, shape (n_maturities, n_strikes)
            Predicted implied volatility surface
        """
        # Create prediction grid
        X_grid = []
        for maturity in maturities:
            for strike in strikes:
                feature = [
                    market_params["S0"],
                    strike,
                    maturity,
                    market_params["r"],
                    market_params["q"],
                    market_params["v0"],
                    market_params["theta"],
                    market_params["kappa"],
                    market_params["sigma"],
                    market_params["rho"],
                ]
                X_grid.append(feature)

        X_grid = np.array(X_grid)

        # Make predictions
        iv_flat = self.predict(X_grid)

        # Reshape to surface
        iv_surface = iv_flat.reshape(len(maturities), len(strikes))

        return iv_surface

    def evaluate_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
        iv_surface_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a complete surface.

        Returns:
        --------
        metrics : dict
            Surface-level evaluation metrics
        """
        # Predict surface
        iv_surface_pred = self.predict_surface(strikes, maturities, market_params)

        # Calculate metrics
        rmse = np.sqrt(
            mean_squared_error(iv_surface_true.flatten(), iv_surface_pred.flatten())
        )
        mae = mean_absolute_error(iv_surface_true.flatten(), iv_surface_pred.flatten())

        # Surface-specific metrics
        surface_rmse = np.sqrt(np.mean((iv_surface_true - iv_surface_pred) ** 2))

        # Smoothness metrics
        smoothness_true = self._calculate_surface_smoothness(iv_surface_true)
        smoothness_pred = self._calculate_surface_smoothness(iv_surface_pred)
        smoothness_error = abs(smoothness_true - smoothness_pred)

        return {
            "rmse": rmse,
            "mae": mae,
            "surface_rmse": surface_rmse,
            "smoothness_error": smoothness_error,
        }

    def _calculate_surface_smoothness(self, surface: np.ndarray) -> float:
        """Calculate surface smoothness metric."""
        # Calculate gradients
        grad_x = np.gradient(surface, axis=1)
        grad_y = np.gradient(surface, axis=0)

        # Smoothness is inverse of gradient magnitude
        smoothness = 1.0 / (1.0 + np.mean(grad_x**2 + grad_y**2))

        return smoothness

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute R² score for regression.

        Parameters:
        -----------
        y_pred : array, shape (n_samples,)
            Predicted values
        y_true : array, shape (n_samples,)
            True values

        Returns:
        --------
        r2 : float
            R² coefficient of determination
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return r2

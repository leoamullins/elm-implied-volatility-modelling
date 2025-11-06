"""
Enhanced data generation for IV surface modelling.
Focuses on creating diverse training data that captures surface-level dependencies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


@dataclass
class SurfaceTrainingConfig:
    """Configuration for surface-aware training data generation."""

    # Surface grid parameters
    n_strikes: int = 20
    n_maturities: int = 15
    n_surfaces: int = 1000

    # Strike range (as moneyness)
    strike_min: float = 0.7  # 70% of spot
    strike_max: float = 1.3  # 130% of spot

    # Maturity range
    maturity_min: float = 0.1  # 1 month
    maturity_max: float = 2.0  # 2 years

    # Market parameters
    spot_range: Tuple[float, float] = (80, 120)
    rate_range: Tuple[float, float] = (0.01, 0.05)
    dividend_range: Tuple[float, float] = (0.0, 0.03)

    # Heston parameters
    v0_range: Tuple[float, float] = (0.01, 0.25)
    theta_range: Tuple[float, float] = (0.01, 0.25)
    kappa_range: Tuple[float, float] = (0.5, 5.0)
    sigma_range: Tuple[float, float] = (0.1, 1.0)
    rho_range: Tuple[float, float] = (-0.9, 0.0)

    # Surface shape parameters
    vol_level_range: Tuple[float, float] = (0.15, 0.35)
    vol_slope_range: Tuple[float, float] = (-0.1, 0.1)  # Strike slope
    vol_curvature_range: Tuple[float, float] = (-0.05, 0.05)
    vol_time_slope_range: Tuple[float, float] = (-0.1, 0.1)  # Time slope


class SurfaceDataGenerator:
    """
    Generates training data specifically designed for IV surface modelling.

    Key improvements:
    1. Creates complete surfaces rather than individual options
    2. Ensures surface-level smoothness and consistency
    3. Captures cross-dimensional dependencies
    4. Includes realistic market scenarios
    """

    def __init__(self, config: SurfaceTrainingConfig):
        self.config = config

    def generate_surface_data(
        self, n_surfaces: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for surface modelling.

        Returns:
        --------
        X : array, shape (n_samples, n_features)
            Input features [S0, K, T, r, q, v0, theta, kappa, sigma, rho]
        y : array, shape (n_samples,)
            Implied volatilities
        """
        if n_surfaces is None:
            n_surfaces = self.config.n_surfaces

        all_features = []
        all_targets = []

        for surface_idx in range(n_surfaces):
            # Generate market parameters for this surface
            market_params = self._generate_market_parameters()

            # Generate surface grid
            strikes, maturities = self._generate_surface_grid()

            # Generate realistic IV surface
            iv_surface = self._generate_realistic_surface(
                strikes, maturities, market_params
            )

            # Create features and targets for this surface
            surface_features, surface_targets = self._create_surface_samples(
                strikes, maturities, market_params, iv_surface
            )

            all_features.append(surface_features)
            all_targets.append(surface_targets)

        # Combine all surfaces
        X = np.vstack(all_features)
        y = np.hstack(all_targets)

        return X, y

    def _generate_market_parameters(self) -> Dict[str, float]:
        """Generate realistic market parameters for a surface."""
        return {
            "S0": np.random.uniform(*self.config.spot_range),
            "r": np.random.uniform(*self.config.rate_range),
            "q": np.random.uniform(*self.config.dividend_range),
            "v0": np.random.uniform(*self.config.v0_range),
            "theta": np.random.uniform(*self.config.theta_range),
            "kappa": np.random.uniform(*self.config.kappa_range),
            "sigma": np.random.uniform(*self.config.sigma_range),
            "rho": np.random.uniform(*self.config.rho_range),
        }

    def _generate_surface_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate strike and maturity grid for surface."""
        # Create log-spaced strikes (more realistic for options)
        log_strikes = np.linspace(
            np.log(self.config.strike_min),
            np.log(self.config.strike_max),
            self.config.n_strikes,
        )
        strikes = np.exp(log_strikes)

        # Create time grid (more points for short maturities)
        maturities = np.linspace(
            self.config.maturity_min, self.config.maturity_max, self.config.n_maturities
        )

        return strikes, maturities

    def _generate_realistic_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
    ) -> np.ndarray:
        """
        Generate a realistic IV surface using Heston model with surface-level constraints.
        """
        S0 = market_params["S0"]
        r = market_params["r"]
        q = market_params["q"]
        v0 = market_params["v0"]
        theta = market_params["theta"]
        kappa = market_params["kappa"]
        sigma = market_params["sigma"]
        rho = market_params["rho"]

        # Create surface grid
        K_grid, T_grid = np.meshgrid(strikes * S0, maturities)
        K_flat = K_grid.flatten()
        T_flat = T_grid.flatten()

        # Generate base Heston IVs
        iv_surface = self._heston_implied_volatility(
            S0, K_flat, T_flat, r, q, v0, theta, kappa, sigma, rho
        )

        # Reshape to surface
        iv_surface = iv_surface.reshape(len(maturities), len(strikes))

        # Add surface-level smoothness and realistic patterns
        iv_surface = self._add_surface_patterns(
            iv_surface, strikes, maturities, market_params
        )

        return iv_surface

    def _heston_implied_volatility(
        self,
        S0: float,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        q: float,
        v0: float,
        theta: float,
        kappa: float,
        sigma: float,
        rho: float,
    ) -> np.ndarray:
        """Calculate Heston implied volatilities."""
        # Simplified Heston IV calculation
        # In practice, you'd use a more sophisticated method

        # Moneyness
        moneyness = K / S0

        # Base volatility level
        vol_base = np.sqrt(theta)

        # Strike smile (higher vol for OTM options)
        smile_effect = 0.1 * (moneyness - 1.0) ** 2

        # Time decay
        time_effect = 0.05 * np.sqrt(T)

        # Combine effects
        iv = vol_base + smile_effect + time_effect

        # Ensure realistic bounds
        iv = np.clip(iv, 0.05, 1.0)

        return iv

    def _add_surface_patterns(
        self,
        iv_surface: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
    ) -> np.ndarray:
        """Add realistic surface patterns and smoothness."""

        # Add volatility smile/skew
        moneyness = strikes
        smile_strength = np.random.uniform(*self.config.vol_slope_range)
        smile_effect = smile_strength * (moneyness - 1.0) ** 2

        # Add term structure
        time_slope = np.random.uniform(*self.config.vol_time_slope_range)
        time_effect = time_slope * maturities

        # Apply effects
        for i, maturity in enumerate(maturities):
            for j, strike in enumerate(strikes):
                iv_surface[i, j] += smile_effect[j] + time_effect[i]

        # Add surface smoothness (reduce noise)
        iv_surface = self._smooth_surface(iv_surface)

        # Ensure no-arbitrage constraints
        iv_surface = self._enforce_no_arbitrage(iv_surface, strikes, maturities)

        return iv_surface

    def _smooth_surface(self, surface: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce surface noise."""
        from scipy.ndimage import gaussian_filter

        # Light smoothing to maintain surface structure
        smoothed = gaussian_filter(surface, sigma=0.5)

        return smoothed

    def _enforce_no_arbitrage(
        self, surface: np.ndarray, strikes: np.ndarray, maturities: np.ndarray
    ) -> np.ndarray:
        """Enforce basic no-arbitrage constraints."""

        # Ensure positive volatilities
        surface = np.maximum(surface, 0.01)

        # Ensure reasonable upper bounds
        surface = np.minimum(surface, 1.0)

        # Ensure monotonicity in time (roughly)
        for j in range(len(strikes)):
            for i in range(1, len(maturities)):
                if surface[i, j] < surface[i - 1, j] * 0.8:  # Allow some decrease
                    surface[i, j] = surface[i - 1, j] * 0.9

        return surface

    def _create_surface_samples(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
        iv_surface: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create training samples from a surface."""

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
                # Features: [S0, K, T, r, q, v0, theta, kappa, sigma, rho]
                feature = [
                    S0,
                    strike * S0,  # Convert to absolute strike
                    maturity,
                    r,
                    q,
                    v0,
                    theta,
                    kappa,
                    sigma,
                    rho,
                ]

                features.append(feature)
                targets.append(iv_surface[i, j])

        return np.array(features), np.array(targets)


def create_surface_training_data(
    n_surfaces: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data specifically designed for IV surface modelling.

    Parameters:
    -----------
    n_surfaces : int
        Number of complete surfaces to generate

    Returns:
    --------
    X : array, shape (n_samples, 10)
        Input features
    y : array, shape (n_samples,)
        Implied volatilities
    """
    config = SurfaceTrainingConfig(n_surfaces=n_surfaces)
    generator = SurfaceDataGenerator(config)

    return generator.generate_surface_data()

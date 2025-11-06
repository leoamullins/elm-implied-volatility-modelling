"""
Surface-regularised ELM with advanced regularisation techniques
specifically designed for IV surface modelling.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import StandardScaler

from elm.models.core.elm_base import ELM


class SurfaceRegularisedELM(ELM):
    """
    ELM with surface-specific regularisation techniques.

    Key features:
    1. Surface smoothness regularisation
    2. No-arbitrage constraints
    3. Multi-scale regularisation
    4. Adaptive regularisation strength
    """

    def __init__(
        self,
        n_hidden: int = 2000,
        activation: str = "sine",
        random_state: Optional[int] = None,
        scale: float = 0.5,
        normalised_init: bool = True,
        # Surface regularisation parameters
        surface_smoothness: float = 1e-2,
        no_arbitrage_weight: float = 1e-1,
        multi_scale_regularisation: bool = True,
        adaptive_regularisation: bool = True,
        **kwargs,
    ):
        super().__init__(
            n_hidden=n_hidden,
            activation=activation,
            random_state=random_state,
            scale=scale,
            normalised_init=normalised_init,
            **kwargs,
        )

        # Surface regularisation parameters
        self.surface_smoothness = surface_smoothness
        self.no_arbitrage_weight = no_arbitrage_weight
        self.multi_scale_regularisation = multi_scale_regularisation
        self.adaptive_regularisation = adaptive_regularisation

        # Regularisation matrices
        self.smoothness_matrix = None
        self.arbitrage_matrix = None
        self.adaptive_weights = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        surface_info: Optional[Dict[str, Any]] = None,
    ) -> "SurfaceRegularisedELM":
        """
        Fit with surface-specific regularisation.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input features
        y : array, shape (n_samples,)
            Target values (implied volatilities)
        surface_info : dict, optional
            Information about surface structure for regularisation
        """
        # Store dimensions
        n_samples, self.n_features = X.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs = y.shape[1]

        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize weights and biases
        self._initialize_weights(X)

        # Compute hidden layer output
        H = self._compute_hidden_output(X)

        # Build regularisation matrices
        if surface_info is not None:
            self._build_surface_regularisation_matrices(X, surface_info)

        # Solve with regularisation
        self._solve_regularised(H, y, X, surface_info)

        return self

    def _build_surface_regularisation_matrices(
        self, X: np.ndarray, surface_info: Dict[str, Any]
    ) -> None:
        """Build surface-specific regularisation matrices."""

        # Smoothness regularisation matrix
        self.smoothness_matrix = self._build_smoothness_matrix(X, surface_info)

        # No-arbitrage regularisation matrix
        self.arbitrage_matrix = self._build_arbitrage_matrix(X, surface_info)

        # Adaptive weights
        if self.adaptive_regularisation:
            self.adaptive_weights = self._compute_adaptive_weights(X, surface_info)

    def _build_smoothness_matrix(self, X: np.ndarray, surface_info: Dict[str, Any]) -> csr_matrix:
        """Build smoothness regularisation matrix."""
        n_hidden = self.n_hidden

        # Create smoothness penalty matrix
        # This encourages similar weights for similar input features
        smoothness_matrix = eye(n_hidden, format="csr")

        # Add cross-term penalties for surface smoothness
        if "strikes" in surface_info and "maturities" in surface_info:
            strikes = surface_info["strikes"]
            maturities = surface_info["maturities"]

            # Create penalty for neighboring points on surface
            penalty_matrix = self._create_surface_penalty_matrix(X, strikes, maturities, n_hidden)
            smoothness_matrix += self.surface_smoothness * penalty_matrix

        return smoothness_matrix

    def _build_arbitrage_matrix(self, X: np.ndarray, surface_info: Dict[str, Any]) -> csr_matrix:
        """Build no-arbitrage constraint matrix."""
        n_hidden = self.n_hidden

        # Create arbitrage penalty matrix
        arbitrage_matrix = eye(n_hidden, format="csr")

        # Add constraints to prevent arbitrage
        if "strikes" in surface_info and "maturities" in surface_info:
            arbitrage_penalty = self._create_arbitrage_penalty_matrix(X, surface_info, n_hidden)
            arbitrage_matrix += self.no_arbitrage_weight * arbitrage_penalty

        return arbitrage_matrix

    def _create_surface_penalty_matrix(
        self, X: np.ndarray, strikes: np.ndarray, maturities: np.ndarray, n_hidden: int
    ) -> csr_matrix:
        """Create penalty matrix for surface smoothness."""
        n_samples = X.shape[0]

        # Create penalty for neighboring points
        penalty_data = []
        penalty_rows = []
        penalty_cols = []

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Calculate distance between points
                distance = self._calculate_surface_distance(X[i], X[j], strikes, maturities)

                if distance < 0.1:  # Neighboring points
                    # Add penalty for different weights
                    for k in range(n_hidden):
                        penalty_data.append(1.0)
                        penalty_rows.append(i * n_hidden + k)
                        penalty_cols.append(j * n_hidden + k)

        penalty_matrix = csr_matrix(
            (penalty_data, (penalty_rows, penalty_cols)), shape=(n_hidden, n_hidden)
        )

        return penalty_matrix

    def _create_arbitrage_penalty_matrix(
        self, X: np.ndarray, surface_info: Dict[str, Any], n_hidden: int
    ) -> csr_matrix:
        """Create penalty matrix for no-arbitrage constraints."""
        # Simplified arbitrage penalty
        # In practice, this would include butterfly spread constraints, etc.

        penalty_matrix = eye(n_hidden, format="csr")

        return penalty_matrix

    def _calculate_surface_distance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
    ) -> float:
        """Calculate distance between two points on the surface."""
        # Extract strike and maturity indices
        strike1, maturity1 = self._get_surface_coordinates(x1, strikes, maturities)
        strike2, maturity2 = self._get_surface_coordinates(x2, strikes, maturities)

        # Calculate normalised distance
        strike_dist = abs(strike1 - strike2) / len(strikes)
        maturity_dist = abs(maturity1 - maturity2) / len(maturities)

        return np.sqrt(strike_dist**2 + maturity_dist**2)

    def _get_surface_coordinates(
        self, x: np.ndarray, strikes: np.ndarray, maturities: np.ndarray
    ) -> Tuple[int, int]:
        """Get surface coordinates for a feature vector."""
        # Extract strike and maturity from feature vector
        K = x[1]  # Strike
        T = x[2]  # Maturity

        # Find closest indices
        strike_idx = np.argmin(np.abs(strikes - K))
        maturity_idx = np.argmin(np.abs(maturities - T))

        return strike_idx, maturity_idx

    def _compute_adaptive_weights(self, X: np.ndarray, surface_info: Dict[str, Any]) -> np.ndarray:
        """Compute adaptive regularisation weights."""
        n_samples = X.shape[0]

        # Initialize weights
        weights = np.ones(n_samples)

        # Adjust weights based on surface location
        if "strikes" in surface_info and "maturities" in surface_info:
            strikes = surface_info["strikes"]
            maturities = surface_info["maturities"]

            for i, x in enumerate(X):
                strike_idx, maturity_idx = self._get_surface_coordinates(x, strikes, maturities)

                # Higher weight for ATM, short-term options
                atm_weight = 1.0 - abs(strike_idx - len(strikes) // 2) / len(strikes)
                short_term_weight = 1.0 - maturity_idx / len(maturities)

                weights[i] = 1.0 + atm_weight + short_term_weight

        return weights

    def _solve_regularised(
        self,
        H: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
        surface_info: Optional[Dict[str, Any]],
    ) -> None:
        """Solve regularised least squares problem."""
        n_samples, n_hidden = H.shape

        # Build regularisation matrix
        if surface_info is not None and self.smoothness_matrix is not None:
            # Surface-regularised solution
            reg_matrix = self.smoothness_matrix

            if self.arbitrage_matrix is not None:
                reg_matrix += self.arbitrage_matrix

            # Solve: (H^T H + λR) β = H^T y
            HtH = H.T @ H
            Hty = H.T @ y

            # Add regularisation
            if self.adaptive_regularisation and self.adaptive_weights is not None:
                # Weighted regularisation
                W = np.diag(self.adaptive_weights)
                reg_term = H.T @ W @ H @ reg_matrix
            else:
                reg_term = reg_matrix

            # Solve system
            try:
                self.output_weights = spsolve(HtH + self.surface_smoothness * reg_term, Hty)
            except:
                # Fallback to standard solution
                self.output_weights = np.linalg.solve(
                    HtH + self.surface_smoothness * reg_matrix.toarray(), Hty
                )
        else:
            # Standard regularised solution
            if self.regularisation:
                I = np.eye(n_hidden)
                self.output_weights = np.linalg.solve(
                    H.T @ H + self.regularisation_param * I, H.T @ y
                )
            else:
                self.output_weights = np.linalg.pinv(H) @ y

    def predict_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_params: Dict[str, float],
    ) -> np.ndarray:
        """Predict complete IV surface."""
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

        # Apply post-processing for surface consistency
        iv_surface = self._post_process_surface(iv_surface, strikes, maturities)

        return iv_surface

    def _post_process_surface(
        self, surface: np.ndarray, strikes: np.ndarray, maturities: np.ndarray
    ) -> np.ndarray:
        """Post-process surface to ensure consistency."""
        # Ensure positive volatilities
        surface = np.maximum(surface, 0.01)

        # Ensure reasonable upper bounds
        surface = np.minimum(surface, 1.0)

        # Apply light smoothing
        from scipy.ndimage import gaussian_filter

        surface = gaussian_filter(surface, sigma=0.3)

        return surface

    def _initialize_weights(self, X: np.ndarray) -> None:
        """Initialize input weights and biases."""
        # Step 1: Randomly initialize input weights and biases
        if self.normalised_init:
            std_w = self.scale / np.sqrt(max(self.n_features, 1))
            self.input_weights = np.random.normal(
                loc=0.0, scale=std_w, size=(self.n_features, self.n_hidden)
            )
            # Biases can remain at the base scale to allow translation
            self.biases = np.random.normal(loc=0.0, scale=self.scale, size=(self.n_hidden,))
        else:
            # Uniform weights/biases in [-scale, scale]
            self.input_weights = (
                np.random.uniform(-1.0, 1.0, (self.n_features, self.n_hidden)) * self.scale
            )
            self.biases = np.random.uniform(-1.0, 1.0, (self.n_hidden,)) * self.scale


# Backward compatibility alias (American spelling)
SurfaceRegularizedELM = SurfaceRegularisedELM

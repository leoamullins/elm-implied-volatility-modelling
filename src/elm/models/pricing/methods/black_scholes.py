import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Union


class BlackScholes:
    """
    Black-Scholes option pricing and implied volatility calculation.
    """

    @staticmethod
    def d1(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> float:
        """Calculate Black-Scholes call option price."""
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
            d2
        )
        return call_price

    @staticmethod
    def put_price(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> float:
        """Calculate Black-Scholes put option price."""
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
            -d1
        )
        return put_price

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate implied volatility using Brent's method.

        Parameters
        ----------
        market_price : float
            Market price of the option
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        q : float
            Dividend yield
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance

        Returns
        -------
        float
            Implied volatility
        """
        if market_price <= 0:
            return 0.0

        # Check for arbitrage bounds
        if option_type.lower() == "call":
            intrinsic_value = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            if market_price < intrinsic_value:
                return 0.0
        else:  # put
            intrinsic_value = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            if market_price < intrinsic_value:
                return 0.0

        def objective(sigma):
            """Objective function for root finding."""
            if option_type.lower() == "call":
                theoretical_price = BlackScholes.call_price(S, K, T, r, sigma, q)
            else:
                theoretical_price = BlackScholes.put_price(S, K, T, r, sigma, q)
            return theoretical_price - market_price

        try:
            # Use Brent's method for root finding
            # Search in reasonable volatility range [0.001, 5.0]
            implied_vol = brentq(
                objective, 0.001, 5.0, xtol=tolerance, maxiter=max_iter
            )
            return float(implied_vol)
        except ValueError:
            # If no solution found, return 0
            return 0.0

    @staticmethod
    def implied_volatility_batch(
        market_prices: Union[list, np.ndarray],
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        option_type: str = "call",
        q: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        """
        Calculate implied volatility for multiple options.

        Parameters
        ----------
        market_prices : array-like
            Market prices of options
        S : float or array-like
            Current stock prices
        K : float or array-like
            Strike prices
        T : float or array-like
            Time to maturity
        r : float or array-like
            Risk-free rates
        option_type : str
            'call' or 'put'
        q : float or array-like
            Dividend yields

        Returns
        -------
        np.ndarray
            Implied volatilities
        """
        market_prices = np.asarray(market_prices)
        S = np.asarray(S)
        K = np.asarray(K)
        T = np.asarray(T)
        r = np.asarray(r)
        q = np.asarray(q)

        # Ensure all arrays have the same length
        n = len(market_prices)

        # Handle scalar or single-element arrays
        try:
            if np.isscalar(S):
                S = np.full(n, float(S))
            elif hasattr(S, "__len__") and len(S) == 1:
                S = np.full(n, float(S[0]))
            elif len(S) != n:
                raise ValueError(
                    f"S length {len(S)} doesn't match market_prices length {n}"
                )
        except TypeError:
            # Handle numpy arrays that don't support len()
            S = np.full(n, float(S))

        try:
            if np.isscalar(K):
                K = np.full(n, float(K))
            elif hasattr(K, "__len__") and len(K) == 1:
                K = np.full(n, float(K[0]))
            elif len(K) != n:
                raise ValueError(
                    f"K length {len(K)} doesn't match market_prices length {n}"
                )
        except TypeError:
            K = np.full(n, float(K))

        try:
            if np.isscalar(T):
                T = np.full(n, float(T))
            elif hasattr(T, "__len__") and len(T) == 1:
                T = np.full(n, float(T[0]))
            elif len(T) != n:
                raise ValueError(
                    f"T length {len(T)} doesn't match market_prices length {n}"
                )
        except TypeError:
            T = np.full(n, float(T))

        try:
            if np.isscalar(r):
                r = np.full(n, float(r))
            elif hasattr(r, "__len__") and len(r) == 1:
                r = np.full(n, float(r[0]))
            elif len(r) != n:
                raise ValueError(
                    f"r length {len(r)} doesn't match market_prices length {n}"
                )
        except TypeError:
            r = np.full(n, float(r))

        try:
            if np.isscalar(q):
                q = np.full(n, float(q))
            elif hasattr(q, "__len__") and len(q) == 1:
                q = np.full(n, float(q[0]))
            elif len(q) != n:
                raise ValueError(
                    f"q length {len(q)} doesn't match market_prices length {n}"
                )
        except TypeError:
            q = np.full(n, float(q))

        implied_vols = np.zeros(n)

        for i in range(n):
            implied_vols[i] = BlackScholes.implied_volatility(
                market_prices[i], S[i], K[i], T[i], r[i], option_type, q[i]
            )

        return implied_vols

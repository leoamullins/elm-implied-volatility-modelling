import numpy as np
from typing import Callable, Tuple, Union
from .black_scholes import BlackScholes


class COSPricer:
    """
    Complete COS (Fourier-Cosine) method for option pricing.
    Reference: Fang and Oosterlee (2008), "A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions"
    """

    def __init__(self, N: int = 256, L: float = 10.0):
        self.N = int(N)
        self.L = float(L)

    def _cumulants_from_cf(
        self, cf: Callable[[complex, float], complex], T: float
    ) -> Tuple[float, float, float]:
        def log_cf(u):
            val = cf(u, T)
            if abs(val) < 1e-15:  # Avoid log(0)
                return -1e10
            return np.log(val)

        eps = 1e-5
        c1 = np.imag((log_cf(eps) - log_cf(0.0)) / eps)

        c2 = -np.real((log_cf(eps) + log_cf(-eps) - 2.0 * log_cf(0.0)) / (eps**2))
        c2 = max(c2, 1e-10)  # Ensure positive

        h = 1e-3
        l0 = log_cf(0.0)
        l1 = log_cf(h)
        lm1 = log_cf(-h)
        l2 = log_cf(2 * h)
        lm2 = log_cf(-2 * h)
        c4 = np.real(l2 + lm2 - 4.0 * (l1 + lm1) + 6.0 * l0) / (h**4)
        c4 = max(c4, 0.0)

        return c1, c2, c4

    def _truncation_range(
        self,
        cf: Callable[[complex, float], complex],
        T: float,
        method: str = "cumulant",
    ) -> Tuple[float, float]:
        if method == "cumulant":
            c1, c2, c4 = self._cumulants_from_cf(cf, T)

            # Truncation range based on cumulants
            width = self.L * np.sqrt(c2)
            a = c1 - width
            b = c1 + width

        elif method == "simple":
            # Simple symmetric range (less robust)
            a = -self.L * np.sqrt(T)
            b = self.L * np.sqrt(T)

        else:
            raise ValueError("method must be 'cumulant' or 'simple'")

        return a, b

    def _cosine_coefficients(
        self, cf: Callable[[complex, float], complex], a: float, b: float, T: float
    ) -> np.ndarray:
        k = np.arange(self.N)
        omega = k * np.pi / (b - a)

        # Evaluate CF at frequencies omega_k
        phi_vals = np.array([cf(u, T) for u in omega])

        # Apply phase shift and take real part
        F_k = np.real(phi_vals * np.exp(-1j * omega * a))

        return F_k

    def _payoff_coefficients_vanilla(
        self, a: float, b: float, K: float, is_call: bool
    ) -> np.ndarray:
        k = np.arange(self.N)
        omega = k * np.pi / (b - a)
        log_K = np.log(K)

        # Chi function
        def chi(c: float, d: float) -> np.ndarray:
            """Integral of exp(x) * cos(ω*(x-a))"""
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = 1.0 + omega**2

                def E(z):
                    return np.exp(z) * (
                        np.cos(omega * (z - a)) + omega * np.sin(omega * (z - a))
                    )

                result = (E(d) - E(c)) / denom
                result[0] = np.exp(d) - np.exp(c)  # Handle k=0 separately

            return result

        # Psi function
        def psi(c: float, d: float) -> np.ndarray:
            """Integral of cos(ω*(x-a))"""
            result = np.zeros(self.N)
            result[0] = d - c  # For k=0

            # For k > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                result[1:] = (1.0 / omega[1:]) * (
                    np.sin(omega[1:] * (d - a)) - np.sin(omega[1:] * (c - a))
                )

            return result

        # Calculate U_k based on option type
        if is_call:
            # Call payoff: max(S - K, 0) = max(exp(x) - K, 0)
            # Integration domain: [log(K), b]
            U_k = (2.0 / (b - a)) * (chi(log_K, b) - K * psi(log_K, b))
        else:
            # Put payoff: max(K - S, 0) = max(K - exp(x), 0)
            # Integration domain: [a, log(K)]
            U_k = (2.0 / (b - a)) * (K * psi(a, log_K) - chi(a, log_K))

        # Apply half-weight to k=0 term (Fourier-cosine series convention)
        U_k[0] *= 0.5

        return U_k

    def price_european_vanilla(
        self,
        K: float,
        T: float,
        r: float,
        cf: Callable[[complex, float], complex],
        is_call: bool = True,
        truncation_method: str = "cumulant",
    ) -> float:
        # Determine truncation range
        a, b = self._truncation_range(cf, T, truncation_method)

        # Compute cosine coefficients
        F_k = self._cosine_coefficients(cf, a, b, T)

        # Compute payoff coefficients
        U_k = self._payoff_coefficients_vanilla(a, b, K, is_call)

        discount_factor = np.exp(-r * T)
        option_price = discount_factor * np.dot(F_k, U_k)

        return float(option_price)

    def price(
        self,
        K: float,
        T: float,
        r: float,
        cf: Callable[[complex, float], complex],
        option_type: str = "call",
        **kwargs,
    ) -> float:
        if option_type.lower() == "call":
            return self.price_european_vanilla(K, T, r, cf, is_call=True)
        else:
            return self.price_european_vanilla(K, T, r, cf, is_call=False)

    def price_batch(
        self,
        strikes: Union[list, np.ndarray],
        T: float,
        r: float,
        cf: Callable[[complex, float], complex],
        option_type: str = "call",
    ) -> np.ndarray:
        strikes = np.asarray(strikes)
        is_call = option_type.lower() == "call"

        # Compute F_k once (same for all strikes)
        a, b = self._truncation_range(cf, T, "cumulant")
        F_k = self._cosine_coefficients(cf, a, b, T)
        discount_factor = np.exp(-r * T)

        # Compute U_k and price for each strike
        prices = np.zeros(len(strikes))
        for i, K in enumerate(strikes):
            U_k = self._payoff_coefficients_vanilla(a, b, K, is_call)
            prices[i] = discount_factor * np.dot(F_k, U_k)

        return prices

    def price_digital(
        self,
        K: float,
        T: float,
        r: float,
        cf: Callable[[complex, float], complex],
        is_call: bool = True,
    ) -> float:
        # Determine truncation range
        a, b = self._truncation_range(cf, T, "cumulant")

        # Compute cosine coefficients
        F_k = self._cosine_coefficients(cf, a, b, T)

        # Digital payoff coefficients
        k = np.arange(self.N)
        omega = k * np.pi / (b - a)
        log_K = np.log(K)

        # Psi function for digital
        def psi(c: float, d: float) -> np.ndarray:
            result = np.zeros(self.N)
            result[0] = d - c
            with np.errstate(divide="ignore", invalid="ignore"):
                result[1:] = (1.0 / omega[1:]) * (
                    np.sin(omega[1:] * (d - a)) - np.sin(omega[1:] * (c - a))
                )
            return result

        if is_call:
            U_k = (2.0 / (b - a)) * psi(log_K, b)
        else:
            U_k = (2.0 / (b - a)) * psi(a, log_K)

        U_k[0] *= 0.5

        # Price
        discount_factor = np.exp(-r * T)
        price = discount_factor * np.dot(F_k, U_k)

        return float(price)

    def get_density(
        self, x_grid: np.ndarray, T: float, cf: Callable[[complex, float], complex]
    ) -> np.ndarray:
        a, b = self._truncation_range(cf, T, "cumulant")
        F_k = self._cosine_coefficients(cf, a, b, T)

        # Recover density from cosine series
        k = np.arange(self.N)
        density = np.zeros(len(x_grid))

        for i, x in enumerate(x_grid):
            cos_terms = np.cos(k * np.pi * (x - a) / (b - a))
            cos_terms[0] *= 0.5  # Half-weight for k=0
            density[i] = (2.0 / (b - a)) * np.dot(F_k, cos_terms)

        return density

    def implied_volatility(
        self,
        market_price: float,
        S0: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Calculate implied volatility from market price using Black-Scholes.

        Parameters
        ----------
        market_price : float
            Market price of the option
        S0 : float
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

        Returns
        -------
        float
            Implied volatility
        """
        return BlackScholes.implied_volatility(
            market_price, S0, K, T, r, option_type, q
        )

    def implied_volatility_batch(
        self,
        market_prices: Union[list, np.ndarray],
        S0: Union[float, np.ndarray],
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
        S0 : float or array-like
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
        return BlackScholes.implied_volatility_batch(
            market_prices, S0, K, T, r, option_type, q
        )

    def price_to_implied_vol(
        self,
        cf: Callable[[complex, float], complex],
        S0: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Calculate implied volatility by first pricing with COS method, then converting to IV.

        This is useful when you want to get the implied volatility that would be extracted
        from a Heston model price using Black-Scholes.

        Parameters
        ----------
        cf : Callable
            Characteristic function
        S0 : float
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

        Returns
        -------
        float
            Implied volatility
        """
        # First get the price using COS method
        heston_price = self.price(K, T, r, cf, option_type, **kwargs)

        # Then convert to implied volatility
        return self.implied_volatility(heston_price, S0, K, T, r, option_type, q)

    def price_to_implied_vol_batch(
        self,
        cf: Callable[[complex, float], complex],
        S0: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        option_type: str = "call",
        q: Union[float, np.ndarray] = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculate implied volatilities for multiple options using COS method.

        Parameters
        ----------
        cf : Callable
            Characteristic function
        S0 : float or array-like
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
        # First get prices using COS method
        heston_prices = self.price_batch(K, T, r, cf, option_type)

        # Then convert to implied volatilities
        return self.implied_volatility_batch(heston_prices, S0, K, T, r, option_type, q)

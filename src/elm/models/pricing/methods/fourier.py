import numpy as np


class FourierPricer:
    def __init__(self, heston_model):
        self.model = heston_model

    def price(
        self,
        K,
        T,
        option_type="call",
        N=4096,
        B=500,
        alpha=1.5,
        **kwargs,
    ):
        K_is_scalar = np.isscalar(K)
        K = np.atleast_1d(K)

        dv = B / N
        dk = 2 * np.pi / (N * dv)

        beta = np.log(self.model.S0) - dk * N / 2

        v = np.arange(N) * dv
        k = beta + np.arange(N) * dk

        if option_type.lower() == "call":
            psi_values = (
                np.exp(-self.model.r * T)
                * self.model.characteristic_function(v - (alpha + 1) * 1j, T)
                / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
            )
        else:
            psi_values = (
                np.exp(-self.model.r * T)
                * self.model.characteristic_function(v - (alpha + 1) * 1j, T)
                / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
            )

        weights = np.ones(N)
        weights[0] = 0.5
        weights *= dv

        x = np.exp(-1j * beta * v) * psi_values * weights

        fft_result = np.fft.fft(x)

        call_values = np.exp(-alpha * k) / np.pi * fft_result.real

        log_K = np.log(K)

        if option_type.lower() == "call":
            prices = np.interp(log_K, k, call_values)
        else:
            call_prices = np.interp(log_K, k, call_values)
            prices = (
                call_prices
                - self.model.S0 * np.exp(-self.model.q * T)
                + K * np.exp(-self.model.r * T)
            )

        return prices[0] if K_is_scalar else prices

import numpy as np


class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r, q=0.0):
        self.S0 = float(S0)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.r = float(r)
        self.q = float(q)

        self.args = {
            "S0": self.S0,
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "r": self.r,
            "q": self.q,
        }
        self.validate_parameters()

    def characteristic_function(self, u: complex, T: float) -> complex:
        kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma, self.rho
        r, q, v0, x0 = self.r, self.q, self.v0, np.log(self.S0)

        iu = 1j * u

        # Standard (risk-neutral) Heston characteristic function of X_T = ln(S_T)
        # Heston (1993), Lord et al. (2008)
        xi = kappa - rho * sigma * iu
        d = np.sqrt(xi * xi + (sigma**2) * (u * u + iu))
        g = (xi - d) / (xi + d)

        exp_neg_dT = np.exp(-d * T)
        one_minus_g = 1.0 - g
        one_minus_g_exp = 1.0 - g * exp_neg_dT

        C = iu * (r - q) * T + (kappa * theta) / (sigma**2) * (
            (xi - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g)
        )

        D = ((xi - d) / (sigma**2)) * ((1.0 - exp_neg_dT) / one_minus_g_exp)

        return np.exp(C + D * v0 + iu * x0)

    def validate_parameters(self):
        bad = []
        if not (-1.0 <= self.rho <= 1.0):
            bad.append("rho (must be in [-1, 1])")
        for key in ["S0", "v0", "kappa", "theta", "sigma"]:
            if getattr(self, key) <= 0:
                bad.append(key)

        if bad:
            raise ValueError(f"Invalid parameters: {', '.join(bad)}.")

            # if self.sigma**2 > 2 * self.kappa * self.theta:
            print("Feller warning.")

        return True

    def get_parameters(self) -> dict:
        return self.args.copy()

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.args:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
            self.args[key] = value
        self.validate_parameters()

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"HestonModel({params_str})"

    def __str__(self) -> str:
        return (
            "Heston Model:\n"
            f"  Stock Price (S0): {self.S0}\n"
            f"  Initial Variance (v0): {self.v0}\n"
            f"  Mean Reversion (κ): {self.kappa}\n"
            f"  Long-run Variance (θ): {self.theta}\n"
            f"  Vol of Vol (σ): {self.sigma}\n"
            f"  Correlation (ρ): {self.rho}\n"
            f"  Risk-free Rate (r): {self.r}\n"
            f"  Dividend Yield (q): {self.q}"
        )

    def copy(self):
        return HestonModel(**self.args.copy())

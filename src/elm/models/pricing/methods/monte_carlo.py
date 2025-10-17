import numpy as np


def print_clean_results(results: dict):
    """Helper to clean Monte Carlo results dictionary."""
    price = results["price"]
    std_error = results["std_error"]
    conf_interval = results["conf_interval"]
    print(
        f"Price: {price:.4f}, Std. Error: {std_error:.4f}, 95% CI: ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})"
    )


class MonteCarlo:
    """Monte Carlo pricer for European options under the Heston model."""

    def __init__(self, heston_model):
        self.model = heston_model

    def simulate_paths(
        self, T, n_steps, n_paths, scheme="euler", antithetic=False, seed=None
    ):
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        n_sim = n_paths // 2 if antithetic else n_paths

        # Initialize arrays
        S = np.zeros((n_sim, n_steps + 1))
        v = np.zeros((n_sim, n_steps + 1))
        S[:, 0] = self.model.S0
        v[:, 0] = self.model.v0

        # When using antithetic variates, pre-store all random draws used by the
        # primary paths so we can reuse them with opposite signs/complements.
        Z1_store = None
        Z2_store = None
        Z_v_store = None
        U_store = None
        if antithetic:
            Z1_store = np.zeros((n_steps, n_sim))
            Z2_store = np.zeros((n_steps, n_sim))
            # For QE scheme we need additional draws. They are harmless if unused.
            Z_v_store = np.zeros((n_steps, n_sim))
            U_store = np.zeros((n_steps, n_sim))

        # Simulation loop
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_sim)
            Z2 = np.random.standard_normal(n_sim)

            if antithetic:
                Z1_store[i, :] = Z1
                Z2_store[i, :] = Z2

            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * (
                self.model.rho * Z1 + np.sqrt(1 - self.model.rho**2) * Z2
            )

            S_curr = S[:, i]
            v_curr = np.maximum(v[:, i], 0)

            if scheme == "euler":
                v[:, i + 1] = (
                    v_curr
                    + self.model.kappa * (self.model.theta - v_curr) * dt
                    + self.model.sigma * np.sqrt(v_curr) * dW2
                )
                v[:, i + 1] = np.maximum(v[:, i + 1], 0)

                S[:, i + 1] = S_curr * np.exp(
                    (self.model.r - self.model.q - 0.5 * v_curr) * dt
                    + np.sqrt(v_curr) * dW1
                )

            elif scheme == "milstein":
                v[:, i + 1] = (
                    v_curr
                    + self.model.kappa * (self.model.theta - v_curr) * dt
                    + self.model.sigma * np.sqrt(v_curr) * dW2
                    + 0.25 * self.model.sigma**2 * (dW2**2 - dt)
                )
                v[:, i + 1] = np.maximum(v[:, i + 1], 0)

                S[:, i + 1] = S_curr * np.exp(
                    (self.model.r - self.model.q - 0.5 * v_curr) * dt
                    + np.sqrt(v_curr) * dW1
                )

            elif scheme == "qe":
                m = self.model.theta + (v_curr - self.model.theta) * np.exp(
                    -self.model.kappa * dt
                )
                s2 = v_curr * self.model.sigma**2 * np.exp(-self.model.kappa * dt) * (
                    1 - np.exp(-self.model.kappa * dt)
                ) / self.model.kappa + self.model.theta * self.model.sigma**2 * (
                    1 - np.exp(-self.model.kappa * dt)
                ) ** 2 / (2 * self.model.kappa)
                psi = s2 / (m**2 + 1e-10)

                mask = psi <= 1.5
                v_next = np.zeros(n_sim)

                # Pre-generate QE draws for all paths; use only where needed
                Z_v_full = np.random.standard_normal(n_sim)
                U_full = np.random.uniform(0.0, 1.0, n_sim)
                if antithetic:
                    Z_v_store[i, :] = Z_v_full
                    U_store[i, :] = U_full

                # For psi <= 1.5
                if np.any(mask):
                    b2 = (
                        2 / psi[mask]
                        - 1
                        + np.sqrt(2 / psi[mask]) * np.sqrt(2 / psi[mask] - 1)
                    )
                    a = m[mask] / (1 + b2)
                    v_next[mask] = a * (np.sqrt(b2) + Z_v_full[mask]) ** 2

                # For psi > 1.5
                if np.any(~mask):
                    p = (psi[~mask] - 1) / (psi[~mask] + 1)
                    beta = (1 - p) / m[~mask]
                    U_slice = U_full[~mask]
                    v_next[~mask] = np.where(
                        U_slice <= p,
                        0,
                        np.log((1 - p) / (1 - U_slice + 1e-10)) / beta,
                    )

                v[:, i + 1] = v_next

                # Stock price update
                K0 = (
                    -self.model.rho
                    * self.model.kappa
                    * self.model.theta
                    * dt
                    / self.model.sigma
                )
                K1 = (
                    0.5
                    * dt
                    * (self.model.kappa * self.model.rho / self.model.sigma - 0.5)
                    - self.model.rho / self.model.sigma
                )
                K2 = (
                    0.5
                    * dt
                    * (self.model.kappa * self.model.rho / self.model.sigma - 0.5)
                    + self.model.rho / self.model.sigma
                )
                K3 = 0.5 * dt * (1 - self.model.rho**2)

                S[:, i + 1] = S_curr * np.exp(
                    (self.model.r - self.model.q) * dt
                    + K0
                    + K1 * v_curr
                    + K2 * v[:, i + 1]
                    + np.sqrt(K3 * (v_curr + v[:, i + 1])) * Z1
                )
            else:
                raise ValueError("scheme must be 'euler', 'milstein', or 'qe'")

        # Antithetic variates: reuse the exact same draws with flipped signs/complements
        if antithetic:
            S_anti = np.zeros((n_sim, n_steps + 1))
            v_anti = np.zeros((n_sim, n_steps + 1))
            S_anti[:, 0] = self.model.S0
            v_anti[:, 0] = self.model.v0

            for i in range(n_steps):
                Z1 = -Z1_store[i, :]
                Z2 = -Z2_store[i, :]

                dW1 = np.sqrt(dt) * Z1
                dW2 = np.sqrt(dt) * (
                    self.model.rho * Z1 + np.sqrt(1 - self.model.rho**2) * Z2
                )

                S_curr = S_anti[:, i]
                v_curr = np.maximum(v_anti[:, i], 0)

                if scheme == "euler":
                    v_anti[:, i + 1] = (
                        v_curr
                        + self.model.kappa * (self.model.theta - v_curr) * dt
                        + self.model.sigma * np.sqrt(v_curr) * dW2
                    )
                    v_anti[:, i + 1] = np.maximum(v_anti[:, i + 1], 0)

                    S_anti[:, i + 1] = S_curr * np.exp(
                        (self.model.r - self.model.q - 0.5 * v_curr) * dt
                        + np.sqrt(v_curr) * dW1
                    )

                elif scheme == "milstein":
                    v_anti[:, i + 1] = (
                        v_curr
                        + self.model.kappa * (self.model.theta - v_curr) * dt
                        + self.model.sigma * np.sqrt(v_curr) * dW2
                        + 0.25 * self.model.sigma**2 * (dW2**2 - dt)
                    )
                    v_anti[:, i + 1] = np.maximum(v_anti[:, i + 1], 0)

                    S_anti[:, i + 1] = S_curr * np.exp(
                        (self.model.r - self.model.q - 0.5 * v_curr) * dt
                        + np.sqrt(v_curr) * dW1
                    )

                elif scheme == "qe":
                    m = self.model.theta + (v_curr - self.model.theta) * np.exp(
                        -self.model.kappa * dt
                    )
                    s2 = v_curr * self.model.sigma**2 * np.exp(
                        -self.model.kappa * dt
                    ) * (
                        1 - np.exp(-self.model.kappa * dt)
                    ) / self.model.kappa + self.model.theta * self.model.sigma**2 * (
                        1 - np.exp(-self.model.kappa * dt)
                    ) ** 2 / (2 * self.model.kappa)
                    psi = s2 / (m**2 + 1e-10)

                    mask = psi <= 1.5
                    v_next = np.zeros(n_sim)

                    # Reuse QE draws with antithetic transforms
                    Z_v_full = -Z_v_store[i, :]
                    U_full = 1.0 - U_store[i, :]

                    if np.any(mask):
                        b2 = (
                            2 / psi[mask]
                            - 1
                            + np.sqrt(2 / psi[mask]) * np.sqrt(2 / psi[mask] - 1)
                        )
                        a = m[mask] / (1 + b2)
                        v_next[mask] = a * (np.sqrt(b2) + Z_v_full[mask]) ** 2

                    if np.any(~mask):
                        p = (psi[~mask] - 1) / (psi[~mask] + 1)
                        beta = (1 - p) / m[~mask]
                        U_slice = U_full[~mask]
                        v_next[~mask] = np.where(
                            U_slice <= p,
                            0,
                            np.log((1 - p) / (1 - U_slice + 1e-10)) / beta,
                        )

                    v_anti[:, i + 1] = v_next

                    K0 = (
                        -self.model.rho
                        * self.model.kappa
                        * self.model.theta
                        * dt
                        / self.model.sigma
                    )
                    K1 = (
                        0.5
                        * dt
                        * (self.model.kappa * self.model.rho / self.model.sigma - 0.5)
                        - self.model.rho / self.model.sigma
                    )
                    K2 = (
                        0.5
                        * dt
                        * (self.model.kappa * self.model.rho / self.model.sigma - 0.5)
                        + self.model.rho / self.model.sigma
                    )
                    K3 = 0.5 * dt * (1 - self.model.rho**2)

                    S_anti[:, i + 1] = S_curr * np.exp(
                        (self.model.r - self.model.q) * dt
                        + K0
                        + K1 * v_curr
                        + K2 * v_anti[:, i + 1]
                        + np.sqrt(K3 * (v_curr + v_anti[:, i + 1])) * Z1
                    )

            S = np.vstack([S, S_anti])
            v = np.vstack([v, v_anti])

        return S, v

    def price(
        self,
        K,
        T,
        option_type="call",
        n_steps=252,
        n_paths=100000,
        scheme="euler",
        antithetic=True,
        seed=None,
        **kwargs,
    ):
        S, v = self.simulate_paths(T, n_steps, n_paths, scheme, antithetic, seed)

        S_T = S[:, -1]

        if option_type.lower() == "call":
            payoffs = np.maximum(S_T - K, 0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discounted_payoffs = np.exp(-self.model.r * T) * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        conf_interval = (price - 1.96 * std_error, price + 1.96 * std_error)

        return {"price": price, "std_error": std_error, "conf_interval": conf_interval}

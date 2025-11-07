from elm.models.core.elm_base import ELM
import numpy as np


class ELMIncremental(ELM):
    """
    Automatically grows the network by adding neurons one at a time
    until the error threshold is reached.

    Based on:
    - Huang et al. (2006) "Convex incremental extreme learning machine"
    - Cheng et al. (2025) Section 2.2
    """

    def __init__(
        self,
        error_threshold=1,
        max_neurons=200,
        initial_neurons=10,
        activation="sigmoid",
        random_state=None,
        scale=1.0,
        verbose=False,
        regularisation=True,
        regularisation_param=1e-3,
        k=1,
    ):
        super().__init__(
            n_hidden=initial_neurons,  # Will grow during training
            activation=activation,
            random_state=random_state,
            scale=scale,
        )

        self.k = k
        self.regularisation = regularisation
        if regularisation:
            self.regularisation_param = regularisation_param
        else:
            self.regularisation_param = 0.0

        self.max_neurons = max_neurons
        self.error_threshold = error_threshold
        self.initial_neurons = initial_neurons
        self.verbose = verbose

        # Track growth history
        self.growth_history = []  # [(n_neurons, rmse), ...]
        self.final_neurons = None
        self.H_history = []
        self.M_history = []
        self.L_history = []
        self.beta = None

        # Initialize lists for incremental weights/biases
        self.weights = []
        self.biases_list = []

    def _compute_hidden_output_with_params(self, X, weights, bias):
        """
        Compute hidden layer output with specified weights and bias.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        weights : array-like, shape (n_features, n_neurons)
            Weight matrix for hidden layer
        bias : array-like, shape (n_neurons,) or scalar
            Bias values for hidden layer

        Returns:
        --------
        H : array-like, shape (n_samples, n_neurons)
            Hidden layer output
        """
        linear_output = X @ weights + bias

        # Apply activation function
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-linear_output))
        elif self.activation == "tanh":
            return np.tanh(linear_output)
        elif self.activation == "relu":
            return np.maximum(0, linear_output)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialise_phase(self, X, y):
        """
        Initialises the ELM with a small number of neurons.
        """
        self.input_weights = np.random.uniform(
            -self.scale, self.scale, size=(self.n_features, self.n_hidden)
        )
        self.biases = np.random.uniform(-self.scale, self.scale, size=(self.n_hidden,))
        H = self._compute_hidden_output(X)

        # Compute initial output weights with regularization
        # beta0 = (H^T H + C*I)^(-1) H^T y
        if self.regularisation:
            identity = np.eye(self.n_hidden)
            self.output_weights = (
                np.linalg.inv(H.T @ H + self.regularisation_param * identity) @ H.T @ y
            )
        else:
            H_pinv = np.linalg.pinv(H)
            self.output_weights = H_pinv @ y

        y_pred = H @ self.output_weights
        score = self.score(y_pred, y)
        self.H_history.append((H, score))
        self.growth_history.append((self.n_hidden, score))

        # Initialize weights and biases lists with initial neurons
        for i in range(self.n_hidden):
            self.weights.append(self.input_weights[:, i : i + 1])
            self.biases_list.append(self.biases[i])

        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularisation: bool = True,
        regularisation_param: float = 1e-3,
    ):
        """
        Train the ELM incrementally by adding neurons until error threshold is met.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training input data
        y : array-like, shape (n_samples, n_outputs)
            Target values
        k : int, default=1
            Number of candidate neurons to try at each step (best one selected)

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        self.regularisation = regularisation
        self.regularisation_param = regularisation_param if regularisation else 0.0

        k = self.k

        n_samples, self.n_features = X.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs = y.shape[1]
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialise an initial phase with a small number of neurons
        self._initialise_phase(X, y)

        # Initial H, M0, L0, beta0
        H0 = self.H_history[-1][0]
        G0 = H0.T @ H0 + self.regularisation_param * np.eye(H0.shape[1])
        M0 = np.linalg.inv(G0)
        L0 = H0.T @ y
        beta0 = M0 @ L0

        self.M_history.append(M0)
        self.L_history.append(L0)
        self.beta = beta0

        # Training loop
        s = 0
        r2 = -np.inf
        while (s < self.max_neurons) and (r2 < self.error_threshold):
            s += 1
            H_old = self.H_history[-1][0]
            M_prev = self.M_history[-1]
            L_prev = self.L_history[-1]

            best_r2 = -np.inf
            best_H, best_beta, best_M, best_L = None, None, None, None
            best_w, best_b = None, None

            # Trying k candidates and selecting the best
            for i in range(k):
                # Candidate weights & bias
                new_weights = np.random.uniform(
                    -self.scale, self.scale, size=(self.n_features, 1)
                )
                new_bias = np.random.uniform(-self.scale, self.scale)

                # Hidden output of candidate
                v_s = self._compute_hidden_output_with_params(
                    X, new_weights, new_bias
                )  # shape (n_samples, 1)

                # Recursive update
                p = H_old.T @ v_s
                q = float(v_s.T @ v_s + self.regularisation_param)

                Delta = float(q - p.T @ M_prev @ p)  # ensures scalar
                Delta_inv = 1.0 / Delta

                top_left = M_prev + (M_prev @ p) @ (Delta_inv * (p.T @ M_prev))
                top_right = -(M_prev @ p) * Delta_inv
                bottom_left = -(p.T @ M_prev) * Delta_inv
                bottom_right = np.array([[Delta_inv]])

                M_s = np.block([[top_left, top_right], [bottom_left, bottom_right]])

                L_s = np.vstack([L_prev, v_s.T @ y])
                beta_s = M_s @ L_s
                if beta_s.ndim == 1:
                    beta_s = beta_s.reshape(-1, 1)

                H_s = np.hstack([H_old, v_s])
                y_pred = H_s @ beta_s

                # R^2 score
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
                r2_i = 1 - ss_res / ss_tot

                # Keep best candidate
                if r2_i > best_r2:
                    best_r2 = r2_i
                    best_H = np.hstack([H_old, v_s])
                    best_beta = beta_s
                    best_M, best_L = M_s, L_s
                    best_w, best_b = new_weights, new_bias

            assert best_H is not None, f"No candidates evaluated (k={k})"
            assert best_beta is not None
            assert best_M is not None
            assert best_L is not None
            assert best_w is not None
            assert best_b is not None

            # Commit best candidate
            self.H_history.append((best_H, best_r2))
            self.M_history.append(best_M)
            self.L_history.append(best_L)
            self.beta = best_beta
            self.weights.append(best_w)
            self.biases_list.append(best_b)
            r2 = best_r2

            # Update growth history
            current_neurons = self.initial_neurons + s
            self.growth_history.append((current_neurons, r2))

            if self.verbose:
                print(f"Step {s:3d} | Hidden neurons: {best_H.shape[1]} | R²: {r2:.4f}")

        # Store final configuration
        self.final_neurons = self.initial_neurons + s
        self.output_weights = self.beta

        # Rebuild input_weights and biases from lists
        self.input_weights = np.hstack(self.weights)
        self.biases = np.array(self.biases_list)
        self.n_hidden = self.final_neurons

        if self.verbose:
            print(
                f"Training complete. Final neurons: {self.final_neurons}, Final R²: {r2:.4f}"
            )

        return self

    def predict(self, X):
        """
        Predict using the trained ELM.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        if self.input_weights is None or self.output_weights is None:
            raise ValueError("Model not trained. Call 'fit' first.")

        # Ensure X is a numpy array
        X = np.array(X)

        # Compute hidden layer output for the input data X
        H = self._compute_hidden_output(X)

        # Make predictions using the learned output weights
        y_pred = H @ self.output_weights
        if self.n_outputs == 1:
            y_pred = y_pred.ravel()

        # Ensure we return a numpy array, not a tuple
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0] if len(y_pred) > 0 else np.array([])

        return np.array(y_pred)

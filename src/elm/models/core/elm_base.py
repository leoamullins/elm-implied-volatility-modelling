"""elm/models/core/elm_base.py - Extreme Learning Machine Base Class."""

import numpy as np


class ELM:
    """
    Extreme Learning Machine (ELM) for regression and classification.

    Based on Huang et al. (2006) and Cheng et al. (2025)
    """

    def __init__(
        self,
        n_hidden,
        activation="sigmoid",
        random_state=None,
        scale=1.0,
        normalised_init: bool = False,
    ):
        """
        Parameters:
        -----------
        n_hidden : int
            Number of hidden layer neurons (L in the paper)
        activation : str
            Activation function: 'sigmoid', 'tanh', 'relu', 'sine'
        random_state : int
            Random seed for reproducibility
        scale : float
            Scaling parameter for random weights initialization
        normalised_init:
            Initialised input weights
        """
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        self.scale = scale
        self.normalised_init = bool(normalised_init)

        # These will be set during fit()
        self.input_weights = None  # W: (n_features, n_hidden)
        self.biases = None  # b: (n_hidden,)
        self.output_weights = None  # β: (n_hidden, n_outputs)
        self.n_features = None
        self.n_outputs = None
        self.regularisataion = None
        self.regularisation_param = None  # C in the paper, degree of regularisation

    def _activation_function(self, X):
        """
        Apply activation function element-wise.

        Parameters:
        -----------
        X : array, shape (n_samples, n_hidden)
            Pre-activation values

        Returns:
        --------
        H : array, shape (n_samples, n_hidden)
            Activated hidden layer output
        """
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-X))

        elif self.activation == "tanh":
            return np.tanh(X)

        elif self.activation == "relu":
            return np.maximum(0, X)  # element-wise

        elif self.activation == "sine":
            return np.sin(X)

        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _compute_hidden_output(self, X, input_weights=None, biases=None):
        """
        Compute hidden layer output H.

        H = g(X @ W + b)

        where:
        - X: input data (n_samples, n_features)
        - W: input weights (n_features, n_hidden)
        - b: biases (n_hidden,)
        - g: activation function

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        H : array, shape (n_samples, n_hidden)
            Hidden layer output
        """

        if input_weights is None:
            input_weights = self.input_weights
        if biases is None:
            biases = self.biases
        linear_output = np.dot(X, input_weights) + biases
        H = self._activation_function(linear_output)

        return H

    def fit(self, X, y, regularisation=False, regularisation_param=1e-3):
        """
        Train the ELM model.

        Steps:
        1. Randomly initialize input weights W and biases b
        2. Compute hidden layer output H = g(XW + b)
        3. Solve for output weights: β = H† @ y (Moore-Penrose inverse)

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Training input data
        y : array, shape (n_samples,) or (n_samples, n_outputs)
            Training target values

        Returns:
        --------
        self : object
            Fitted ELM model
        """

        # Storing dimensions
        n_samples, self.n_features = X.shape

        # Handle 1D data
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs = y.shape[1]

        # Set a random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Step 1: Randomly initialize input weights and biases
        # W ~ Normal(0, scale)
        # If normalized_init is True, use variance-aware Normal init with
        # std = scale / sqrt(n_features) so Var(XW) scales well with feature dimension
        if self.normalised_init:
            std_w = self.scale / np.sqrt(max(self.n_features, 1))
            self.input_weights = np.random.normal(
                loc=0.0, scale=std_w, size=(self.n_features, self.n_hidden)
            )
            # Biases can remain at the base scale to allow translation
            self.biases = np.random.normal(
                loc=0.0, scale=self.scale, size=(self.n_hidden,)
            )
        else:
            # Uniform weights/biases in [-scale, scale]
            self.input_weights = (
                np.random.uniform(-1.0, 1.0, (self.n_features, self.n_hidden))
                * self.scale
            )
            self.biases = np.random.uniform(-1.0, 1.0, (self.n_hidden,)) * self.scale

        # Step 2: Compute hidden layer output
        H = self._compute_hidden_output(X)  # (n_samples, n_hidden)

        # Step 3: Solve for output weights using Moore-Penrose inverse
        # beta = H_pinv @ y
        if not regularisation:
            self.regularisataion = regularisation
            H_pinv = np.linalg.pinv(H)
            self.output_weights = H_pinv @ y  # (n_hidden, n_outputs)
        elif regularisation:
            self.regularisataion = regularisation
            self.regularisation_param = regularisation_param  # default value

            # Use more stable formulation based on data size
            # If n_samples < n_hidden, use dual form for better conditioning
            if n_samples < self.n_hidden:
                # Dual form: beta = H^T (HH^T + lambda*I)^(-1) y
                I_n = np.eye(n_samples)
                inv_term = np.linalg.solve(H @ H.T + self.regularisation_param * I_n, y)
                self.output_weights = H.T @ inv_term
            else:
                # Primal form: beta = (H^T H + lambda*I)^(-1) H^T y
                I_L = np.eye(self.n_hidden)
                self.output_weights = np.linalg.solve(
                    H.T @ H + self.regularisation_param * I_L, H.T @ y
                )
        else:
            raise ValueError("regularisation must be True or False")

        return self

    def predict(self, X):
        """
        Predict using the trained ELM model.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input data for prediction

        Returns:
        --------
        y_pred : array, shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values
        """
        if self.input_weights is None or self.output_weights is None:
            raise ValueError("Model not trained. Call 'fit' first.")

        # Compute hidden layer output
        H = self._compute_hidden_output(X)  # (n_samples, n_hidden)

        # Compute predictions
        y_pred = np.dot(H, self.output_weights)  # (n_samples, n_outputs)

        # If single output, return 1D array
        if self.n_outputs == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def score(self, y_pred, y):
        """
        Compute R² score for regression (or accuracy for classification).

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Test input data
        y : array, shape (n_samples,)
            True target values

        Returns:
        --------
        score : float
            R² coefficient of determination
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred must have the same shape")

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return r2


class ELMClassifier(ELM):
    """
    ELM for classification tasks.
    Uses one-hot encoding for targets.
    """

    def fit(self, X, y):
        """
        Fit the ELM classifier.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Training input data
        y : array, shape (n_samples,)
            Training class labels

        Returns:
        -----------
        self : object
            Fitted ELM classifier
        """
        self.classes_ = np.unique(y)
        y_onehot = self._to_onehot(y)

        # Call parent fit method
        super().fit(X, y_onehot)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        y_pred_scores = super().predict(X)  # Get raw scores
        y_pred = np.argmax(y_pred_scores, axis=1)  # Class with highest score
        return self.classes_[y_pred]

    def predict_probability(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Training input data

        Returns:
        -----------
        probs : array, shape (n_samples, n_classes)
            Class probabilities for each sample from the
            Fitted ELM classifier
        """
        y_pred = super().predict(X)

        # Softmax
        exp_scores = np.exp(
            y_pred - np.max(y_pred, axis=1, keepdims=True)
        )  # subtract max for numerical stability
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def _to_onehot(self, y):
        """
        Convert class labels to one-hot encoding.

        Parameters:
        -----------
        y : array, shape (n_samples,)
            Class labels

        Returns:
        -----------
        y_onehot : array, shape (n_samples, n_classes)
            One-hot encoded class labels
        """
        n_classes = len(self.classes_)
        y_onehot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            class_idx = np.where(self.classes_ == label)[0][0]
            y_onehot[i, class_idx] = 1
        return y_onehot

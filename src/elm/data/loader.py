from typing import Union, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from pathlib import Path

from elm.models.pricing.elm_pricer import generate_heston_training_data


def load_training_data(
    cache_dir: Optional[str | Path] = None,
    n_samples: int = 10000,
    option_type: Literal["call", "put"] = "call",
    pricing_method: Literal["cos", "fourier", "monte_carlo"] = "cos",
    random_state: Optional[int] = None,
    parameter_ranges: Optional[dict] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data (X, y) arrays from disk cache or generate new ones.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to save/load the CSV files. If None, generates data without caching.
    n_samples : int
        Number of samples to generate (if not cached)
    option_type : str
        'call' or 'put'
    pricing_method : str
        Method to compute ground truth: 'cos', 'fourier', 'monte_carlo'
    random_state : int, optional
        Random seed for reproducibility
    parameter_ranges : dict, optional
        Custom ranges for parameters
    force_recompute : bool
        If True, regenerate data even if cache exists

    Returns
    -------
    X : np.ndarray, shape (n_samples, 10)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Target option prices
    """

    # In-memory only path
    if cache_dir is None:
        print("No cache directory provided. Generating data in-memory...")
        return generate_heston_training_data(
            n_samples=n_samples,
            option_type=option_type,
            pricing_method=pricing_method,
            random_state=random_state,
            parameter_ranges=parameter_ranges,
        )

    # Setup cache paths
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create stable filename based on parameters
    filename_parts = [
        "training_data",
        f"n{n_samples}",
        f"{option_type}",
        f"{pricing_method}",
    ]
    if random_state is not None:
        filename_parts.append(f"seed{random_state}")

    base_filename = "_".join(filename_parts)
    X_path = cache_path / f"{base_filename}_X.csv"
    y_path = cache_path / f"{base_filename}_y.csv"

    # Load from cache if present and not forcing recompute
    if X_path.exists() and y_path.exists() and not force_recompute:
        print(f"Loading cached training data from {cache_path}...")
        X = pd.read_csv(X_path).values
        y = pd.read_csv(y_path).values.flatten()
        print(f"Loaded X: {X.shape}, y: {y.shape}")
        return X, y

    # Generate new data
    print(f"Generating new training data ({n_samples} samples)...")
    X, y = generate_heston_training_data(
        n_samples=n_samples,
        option_type=option_type,
        pricing_method=pricing_method,
        random_state=random_state,
        parameter_ranges=parameter_ranges,
    )

    # Save to cache
    print(f"Saving training data to {cache_path}...")
    pd.DataFrame(X).to_csv(X_path, index=False)
    pd.DataFrame(y, columns=["price"]).to_csv(y_path, index=False)
    print(f"Saved X: {X.shape} -> {X_path.name}")
    print(f"Saved y: {y.shape} -> {y_path.name}")

    return X, y

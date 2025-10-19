from elm.models.pricing.elm_pricer import OptionPricingELM, create_train_val_test_split
from elm.data.loader import load_training_data
from sklearn.metrics import root_mean_squared_error
import numpy as np
import time

X, y = load_training_data(n_samples=100000, cache_dir="data/", random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
    X, y, random_state=42, val_size=0.0, test_size=0.2, train_size=0.8
)

model = OptionPricingELM(
    n_hidden=3000,
    activation="sine",
    scale=0.5,
    random_state=42,
    normalise_features=True,
    normalise_target=False,
    regularisation_param=1e-3,
    clip_negative=True,
    target_transform="none",
    forward_normalise=True,
    normalised_init=True,
)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time


y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

# Compare with analytical methods
elm_prices, cos_prices = model.compare_with_analytical(
    X_test, method="cos", comparison_mode="implied_volatility"
)
correlation = np.corrcoef(elm_prices, cos_prices)[0, 1]
price_diff = np.mean(np.abs(elm_prices - cos_prices))

print("ELM Options IV Benchmarking Results:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training time: {training_time:.4f} seconds")
print(f"Samples per second: {len(X_train) / training_time:.0f}")
print()
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2: {r2:.4f}")
print()
print(f"Correlation with COS method: {correlation:.6f}")
print(f"Mean absolute implied volatility difference: {price_diff:.6f}")
print()
print(f"Target value range: [{y_test.min():.4f}, {y_test.max():.4f}]")
print(f"Sample targets: {y_test[:5]}")

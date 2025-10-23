import pandas as pd
from sklearn.metrics import root_mean_squared_error

from elm.data.loader import load_training_data
from elm.models.pricing.elm_pricer import (
    OptionPricingELM,
    create_train_val_test_split,
)

X, y = load_training_data(n_samples=100000, cache_dir="data/")

X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
    X, y, random_state=42, val_size=0.0, test_size=0.2, train_size=0.8
)

model = OptionPricingELM(
    n_hidden=3000,
    activation="sine",
    scale=1,
    random_state=42,
    normalise_features=True,
    normalise_target=False,
    regularisation_param=1e-3,  # Strong regularization prevents overfitting
    clip_negative=False,  # Ensure no negative option prices
    target_transform="none",
    forward_normalise=True,
    normalised_init=True,
)


model.fit(X_train, y_train)
# model.save_model("models/elmtrainedsyntheticdata.pkl")

# Storing trained model
# input_weights = pd.Series(model.input_weights.flatten())
# input_weights.to_csv("data/input_weights.csv")
# output_weights = pd.Series(model.output_weights.flatten())
# output_weights.to_csv("data/output_weights.csv")
# biases = pd.Series(model.biases.flatten())
# biases.to_csv("data/biases.csv")
y_pred = model.predict(X_test)

print(y_pred)

print(root_mean_squared_error(y_true=y_test, y_pred=y_pred))

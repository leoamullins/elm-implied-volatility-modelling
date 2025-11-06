import joblib
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from elm.data.loader import load_training_data
from elm.models.pricing.elm_pricer import OptionPricingELM

X, y = load_training_data(n_samples=250000)

X_train, X_test, y_train, y_test = train_test_split(X, y)

op = OptionPricingELM(
    n_hidden=4000,
    activation="sine",
    scale=0.5,
    random_state=42,
    normalise_features=True,
    normalise_target=False,
    regularisation_param=1e-3,  # Strong regularization prevents overfitting
    clip_negative=False,  # Ensure no negative option prices
    target_transform="none",
    forward_normalise=True,
    normalised_init=True,
)

op.fit(X_train, y_train)

y_pred = op.predict(X_test)

joblib.dump(op, "models/elmtraininglargedataset.pkl")

print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")

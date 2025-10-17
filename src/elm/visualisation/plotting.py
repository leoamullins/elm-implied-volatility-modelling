from elm.models.pricing.elm_pricer import OptionPricingELM, create_train_val_test_split
from elm.data.loader import load_training_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error

X, y = load_training_data(n_samples=100000, cache_dir="data/")

X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
    X, y, random_state=42, val_size=0.0, test_size=0.2, train_size=0.8
)


def plotting_scale_neurons(X_train, y_train, X_test, y_test, n_seeds: int = 5):
    fig = plt.figure(figsize=(10, 4))
    (left, right) = fig.subfigures(1, 2, wspace=0.05)

    left_ax = left.subplots(1, 1)
    right_ax = right.subplots(1, 1)

    # Left plot: RMSE vs Scale for different numbers of neurons
    scale_range = np.linspace(0.2, 1.5, 10, endpoint=True)
    L_values = [1000, 2000, 3000]
    seeds = list(range(n_seeds))

    print("\n=== RMSE vs Scale (Averaged over Seeds) ===")
    for l_idx, L in enumerate(L_values, 1):
        xl, yl = [], []
        for s_idx, scale in enumerate(scale_range, 1):
            errs = []
            print(f"\nL={L}, Scale={scale:.2f} ... ", end="")
            for seed in seeds:
                model = OptionPricingELM(
                    n_hidden=L,
                    activation="sine",
                    scale=scale,
                    random_state=seed,
                    normalise_features=True,
                    normalise_target=False,
                    regularisation_param=1e-3,
                    clip_negative=False,  # disable for fair RMSE comparison
                    normalised_init=True,
                    target_transform="none",
                    forward_normalise=True,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                err = root_mean_squared_error(y_test, y_pred)
                errs.append(err)
            mean_rmse = np.mean(errs)
            yl.append(mean_rmse)
            xl.append(scale)
            print(f"mean RMSE = {mean_rmse:.6f}")
        left_ax.plot(xl, yl, marker="o", label=f"L={L}")

    left_ax.set_xlabel("Scale")
    left_ax.set_ylabel("Mean RMSE (across seeds)")
    left_ax.set_title("RMSE vs Scale (averaged)")
    left_ax.legend()
    left_ax.grid(True, alpha=0.3)

    # Right plot: RMSE vs Neurons for different scales
    neurons_range = np.linspace(400, 4000, 10, endpoint=True, dtype=int)
    scale_values = [0.5, 1.0, 1.5]

    print("\n=== RMSE vs Neurons (Averaged over Seeds) ===")
    for scale_idx, scale in enumerate(scale_values, 1):
        xn, yn = [], []
        for n_idx, n in enumerate(neurons_range, 1):
            errs = []
            print(f"\nScale={scale}, L={n} ... ", end="")
            for seed in seeds:
                model = OptionPricingELM(
                    n_hidden=int(n),
                    activation="sine",
                    scale=scale,
                    random_state=seed,
                    normalise_features=True,
                    normalise_target=False,
                    regularisation_param=1e-3,
                    clip_negative=False,
                    normalised_init=True,
                    target_transform="none",
                    forward_normalise=True,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                err = root_mean_squared_error(y_test, y_pred)
                errs.append(err)
            mean_rmse = np.mean(errs)
            xn.append(n)
            yn.append(mean_rmse)
            print(f"mean RMSE = {mean_rmse:.6f}")
        right_ax.plot(xn, yn, marker="o", label=f"Scale={scale}")

    right_ax.set_xlabel("Number of Neurons")
    right_ax.set_ylabel("Mean RMSE (across seeds)")
    right_ax.set_title("RMSE vs Neurons (averaged)")
    right_ax.legend()
    right_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plotting_scale_neurons(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

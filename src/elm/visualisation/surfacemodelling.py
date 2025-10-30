from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from elm.models.pricing.elm_pricer import generate_heston_training_data


@dataclass
class HESTON_PARAMS:
    v0 : float # Initial Variance
    theta: float # Long-term variance
    kappa: float # Mean reversion
    sigma: float # Volatility
    rho: float # Correlation
    
DEFAULT_NUMBERS = {"strike_range_count": 100, "time_range_count": 100}
hp = HESTON_PARAMS(0.04, 0.04, 2.0, 0.3, -0.7)

"""
    Expects a model that calculates implied volatilities!!
"""
def plot_implied_volatility_surface(
    model, S0, strikes_range, time_range, r, q, heston_params, save_path
):
    
    # Setup and Grid Generation
    strike_range_count = DEFAULT_NUMBERS["strike_range_count"]
    time_range_count = DEFAULT_NUMBERS["time_range_count"]
    strikes = np.linspace(min(strikes_range), max(strikes_range), strike_range_count)
    times = np.linspace(min(time_range), max(time_range), time_range_count)
    
    K_grid, T_grid = np.meshgrid(strikes, times)
    
    # Create feature matrix for ELM
    n_points = strike_range_count * time_range_count
    X_surface = np.zeros((n_points, 10))
    
    i = 0
    for T in times:
        for K in strikes:
            X_surface[i] = [
                S0,
                K,
                T,
                r,
                q,
                heston_params.v0,
                heston_params.theta,
                heston_params.kappa,
                heston_params.sigma,
                heston_params.rho,
            ]
            i += 1
          
    elm_iv = model.predict(X_surface)
    iv_grid = np.reshape(elm_iv, (time_range_count, strike_range_count))

    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(20, 12))

    # Panel 1: 3D Surface Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surface = ax1.plot_surface(K_grid, T_grid, iv_grid, cmap='viridis')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Time to Maturity')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('ELM Implied Volatility Surface')
    fig.colorbar(surface, ax=ax1, shrink=0.5)

    # Panel 2: 2D Heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    heatmap = ax2.imshow(iv_grid, aspect='auto', cmap='viridis', origin='lower',
                         extent=(strikes.min(), strikes.max(), times.min(), times.max()))
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Time to Maturity')
    ax2.set_title('IV Heatmap')
    fig.colorbar(heatmap, ax=ax2)

    # Panel 3: Volatility Smile/Skew
    ax3 = fig.add_subplot(2, 3, 3)
    moneyness = strikes / S0
    for time_index in range(0, len(times), max(1, len(times) // 4)):  # Every 4th maturity
        T = times[time_index]
        ax3.plot(moneyness, iv_grid[time_index, :], label=f'T={T:.1f}')
    ax3.set_xlabel('Moneyness (K/S0)')
    ax3.set_ylabel('Implied Volatility')
    ax3.set_title('IV Smile/Skew')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: ATM Term Structure
    ax4 = fig.add_subplot(2, 3, 4)
    atm_index = int(np.argmin(np.abs(strikes - S0)))  # Nearest strike to S0 (ATM)
    ax4.plot(times, iv_grid[:, atm_index], 'b-o', markersize=3)
    ax4.set_xlabel('Time to Maturity')
    ax4.set_ylabel('ATM Implied Volatility')
    ax4.set_title('ATM Term Structure')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Volatility Smiles at Different Times
    ax5 = fig.add_subplot(2, 3, 5)
    smile_times = [0.25, 0.5, 1.0, 2.0]
    for target_T in smile_times:
        closest_idx = np.argmin(np.abs(times - target_T))
        ax5.plot(strikes, iv_grid[closest_idx, :], label=f'T={times[closest_idx]:.2f}')
    ax5.set_xlabel('Strike Price')
    ax5.set_ylabel('Implied Volatility')
    ax5.set_title('Volatility Smiles')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    valid_iv = iv_grid[iv_grid > 0]

    def _fmt_stat(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

    if valid_iv.size > 0:
        mean_iv = float(np.mean(valid_iv))
        median_iv = float(np.median(valid_iv))
        std_iv = float(np.std(valid_iv))
        min_iv = float(np.min(valid_iv))
        max_iv = float(np.max(valid_iv))
    else:
        mean_iv = median_iv = std_iv = min_iv = max_iv = float("nan")

    stats_text = f"""
    Implied Volatility Statistics:

    Mean IV: {_fmt_stat(mean_iv)}
    Median IV: {_fmt_stat(median_iv)}
    Std Dev: {_fmt_stat(std_iv)}
    Min IV: {_fmt_stat(min_iv)}
    Max IV: {_fmt_stat(max_iv)}
    
    Surface Parameters:
    Strike Range: [{strikes.min():.2f}, {strikes.max():.2f}]
    Time Range: [{times.min():.2f}, {times.max():.2f}]
    Spot Price (S0): {S0:.2f}
    Risk-free Rate (r): {r:.4f}
    Dividend Yield (q): {q:.4f}
    
    Heston Parameters:
    v0 (Initial Var): {heston_params.v0:.4f}
    theta (Long-term Var): {heston_params.theta:.4f}
    kappa (Mean Reversion): {heston_params.kappa:.4f}
    sigma (Vol of Vol): {heston_params.sigma:.4f}
    rho (Correlation): {heston_params.rho:.4f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace')

    plt.tight_layout()

    # Step 6: Save and Display
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return iv_grid, strikes, times

if __name__ == "__main__":
    # Import necessary modules
    from sklearn.model_selection import train_test_split

    from elm.models.pricing.elm_pricer import OptionPricingELM
    
    # Generate training data
    print("Generating training data...")
    X, y = generate_heston_training_data(n_samples=10000, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train ELM model
    print("Training ELM model...")
    model = OptionPricingELM(
        n_hidden=3000,
        activation="sine",
        scale=0.5,
        random_state=42,
        normalise_features=True,
        normalise_target=False,
        regularisation_param=1e-3,
        clip_negative=False,
        target_transform="none",
        forward_normalise=True,
        normalised_init=True
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    print(f"Model trained. RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.4f}")
    
    # Set parameters for implied volatility surface
    S0 = 100.0
    strikes_range = [80, 120]
    time_range = [0.25, 2.0]
    r = 0.05
    q = 0.0
    
    # Plot the implied volatility surface
    print("Plotting implied volatility surface...")
    plot_implied_volatility_surface(
        model=model,
        S0=S0,
        strikes_range=strikes_range,
        time_range=time_range,
        r=r,
        q=q,
        heston_params=hp,
        save_path="implied_volatility_surface.png"
    )

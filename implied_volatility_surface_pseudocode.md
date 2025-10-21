# Implied Volatility Surface Generation - Pseudocode

## Overview
Generate and visualize implied volatility surfaces from ELM option pricing models.

## Main Algorithm

```
FUNCTION plot_implied_volatility_surface(model, S0, strikes_range, time_range, r, q, heston_params, save_path):
    
    // Step 1: Setup and Grid Generation
    strikes = linspace(strikes_range.min, strikes_range.max, strikes_range.count)
    times = linspace(time_range.min, time_range.max, time_range.count)
    K_grid, T_grid = meshgrid(strikes, times)
    
    // Step 2: Create Feature Matrix for ELM
    n_points = strikes_range.count * time_range.count
    X_surface = zeros(n_points, 10)
    
    FOR each time T in times:
        FOR each strike K in strikes:
            X_surface[idx] = [
                S0,                    // Spot price
                K,                     // Strike price  
                T,                     // Time to maturity
                r,                     // Risk-free rate
                q,                     // Dividend yield
                heston_params.v0,      // Initial variance
                heston_params.theta,   // Long-term variance
                heston_params.kappa,  // Mean reversion
                heston_params.sigma,   // Vol of vol
                heston_params.rho      // Correlation
            ]
            idx++
    
    // Step 3: ELM Prediction
    elm_prices = model.predict(X_surface)
    price_grid = reshape(elm_prices, time_range.count, strikes_range.count)
    
    // Step 4: Calculate Implied Volatilities
    iv_grid = zeros_like(price_grid)
    
    FOR i = 0 to time_range.count:
        FOR j = 0 to strikes_range.count:
            IF price_grid[i,j] > 0:
                iv = implied_volatility_black_scholes(
                    price_grid[i,j], S0, strikes[j], times[i], r, q, "call"
                )
                iv_grid[i,j] = iv IF not isnan(iv) ELSE 0
    
    // Step 5: Create Visualization
    fig = create_figure(2, 3, size=(20, 12))
    
    // Panel 1: 3D Surface Plot
    ax1 = fig.subplot(2, 3, 1, projection='3d')
    surface = ax1.plot_surface(K_grid, T_grid, iv_grid, colormap='viridis')
    ax1.set_labels('Strike Price', 'Time to Maturity', 'Implied Volatility')
    ax1.set_title('ELM Implied Volatility Surface')
    
    // Panel 2: 2D Heatmap
    ax2 = fig.subplot(2, 3, 2)
    heatmap = ax2.imshow(iv_grid, colormap='viridis')
    ax2.set_labels('Strike Price', 'Time to Maturity')
    ax2.set_title('IV Heatmap')
    
    // Panel 3: Volatility Smile/Skew
    ax3 = fig.subplot(2, 3, 3)
    moneyness = strikes / S0
    FOR each time T in times[::4]:  // Every 4th maturity
        ax3.plot(moneyness, iv_grid[time_index, :], label=f'T={T:.1f}')
    ax3.set_labels('Moneyness (K/S0)', 'Implied Volatility')
    ax3.set_title('IV Smile/Skew')
    ax3.legend()
    
    // Panel 4: ATM Term Structure
    ax4 = fig.subplot(2, 3, 4)
    atm_index = strikes_range.count // 2  // Middle strike (ATM)
    ax4.plot(times, iv_grid[:, atm_index], 'b-o')
    ax4.set_labels('Time to Maturity', 'ATM Implied Volatility')
    ax4.set_title('ATM Term Structure')
    
    // Panel 5: Volatility Smiles at Different Times
    ax5 = fig.subplot(2, 3, 5)
    smile_times = [0.25, 0.5, 1.0, 2.0]
    FOR each target_T in smile_times:
        closest_idx = argmin(abs(times - target_T))
        ax5.plot(strikes, iv_grid[closest_idx, :], label=f'T={times[closest_idx]:.2f}')
    ax5.set_labels('Strike Price', 'Implied Volatility')
    ax5.set_title('Volatility Smiles')
    ax5.legend()
    
    // Panel 6: Statistics
    ax6 = fig.subplot(2, 3, 6)
    ax6.axis('off')
    valid_iv = iv_grid[iv_grid > 0]
    stats = calculate_statistics(valid_iv, strikes_range, time_range, S0, r, q, heston_params)
    ax6.text(stats)
    
    // Step 6: Save and Display
    IF save_path:
        save_figure(fig, save_path, dpi=300)
    
    display(fig)
    RETURN iv_grid, strikes, times

END FUNCTION
```

## Implied Volatility Calculation

```
FUNCTION implied_volatility_black_scholes(option_price, S0, K, T, r, q, option_type):
    
    bs = BlackScholes()
    
    FUNCTION objective(sigma):
        TRY:
            IF option_type == "call":
                model_price = bs.call_price(S0, K, T, r, q, sigma)
            ELSE:
                model_price = bs.put_price(S0, K, T, r, q, sigma)
            RETURN (model_price - option_price)²
        CATCH:
            RETURN 1e10
    
    TRY:
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        RETURN result.x IF result.success ELSE NaN
    CATCH:
        RETURN NaN

END FUNCTION
```

## Statistics Calculation

```
FUNCTION calculate_statistics(valid_iv, strikes_range, time_range, S0, r, q, heston_params):
    
    stats = {
        mean_iv: mean(valid_iv),
        std_iv: std(valid_iv),
        min_iv: min(valid_iv),
        max_iv: max(valid_iv),
        strike_range: f"{strikes_range.min} - {strikes_range.max}",
        time_range: f"{time_range.min} - {time_range.max} years",
        spot_price: S0,
        risk_free_rate: f"{r:.1%}",
        dividend_yield: f"{q:.1%}",
        heston_v0: heston_params.v0,
        heston_theta: heston_params.theta,
        heston_kappa: heston_params.kappa,
        heston_sigma: heston_params.sigma,
        heston_rho: heston_params.rho
    }
    
    RETURN format_statistics_text(stats)

END FUNCTION
```

## Main Execution Flow

```
MAIN:
    // Load and preprocess data
    df = load_csv("data/options_data_20k.csv")
    df_clean = data_preprocessing(df)
    
    // Create feature matrix
    X = create_feature_matrix(df_clean)
    y = df_clean["trade_iv"]
    
    // Train ensemble of ELM models
    trained_models, predictions, X_test, y_test, ensemble_pred, uncertainty = train_model(X, y)
    
    // Evaluate model performance
    evaluate_model(y_test, ensemble_pred, uncertainty)
    
    // Create comprehensive plots
    plot_results(y_test, ensemble_pred, uncertainty, save_path="elm_option_pricing_results.png")
    
    // Generate implied volatility surface
    IF trained_models:
        print("IMPLIED VOLATILITY SURFACE GENERATION")
        
        iv_model = trained_models[0]  // Use first model
        
        iv_grid, strikes, times = plot_implied_volatility_surface(
            model=iv_model,
            S0=100,
            strikes_range=(80, 120, 21),
            time_range=(0.1, 2.0, 20),
            r=0.05,
            q=0.02,
            save_path="elm_implied_volatility_surface.png"
        )

END MAIN
```

## Key Parameters

```
PARAMETERS:
    S0: Spot price (default: 100)
    strikes_range: (min_strike, max_strike, n_strikes) (default: (80, 120, 21))
    time_range: (min_time, max_time, n_times) (default: (0.1, 2.0, 20))
    r: Risk-free rate (default: 0.05)
    q: Dividend yield (default: 0.02)
    heston_params: {
        v0: Initial variance (default: 0.04)
        theta: Long-term variance (default: 0.04)
        kappa: Mean reversion (default: 2.0)
        sigma: Vol of vol (default: 0.3)
        rho: Correlation (default: -0.7)
    }
```

## Output Files

```
OUTPUTS:
    - elm_option_pricing_results.png: Model evaluation plots
    - elm_implied_volatility_surface.png: IV surface visualization
```

## Visualization Panels

```
PANELS:
    1. 3D Surface: Interactive 3D implied volatility surface
    2. Heatmap: 2D color-coded IV surface
    3. Smile/Skew: Volatility smile across maturities
    4. Term Structure: ATM volatility over time
    5. Volatility Smiles: Multiple smiles at different times
    6. Statistics: Comprehensive parameter and performance stats
```

This pseudocode provides a complete blueprint for implementing the implied volatility surface generation functionality with your ELM option pricing models.

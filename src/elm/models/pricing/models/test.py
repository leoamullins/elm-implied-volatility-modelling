import QuantLib as ql
import numpy as np

# --- Market parameters ---
S0 = 100.0  # spot
r = 0.05  # risk-free rate
q = 0.00  # dividend yield
T = 1.0  # maturity (years)
K = 100.0  # strike

# --- Heston parameters ---
v0 = 0.04  # initial variance
kappa = 2  # mean reversion speed
theta = 0.05  # long-term variance
sigma = 0.3  # vol of vol
rho = -0.5  # correlation

# --- QuantLib setup ---
calculation_date = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = calculation_date

spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
rf_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, r, ql.Actual365Fixed())
)
div_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, q, ql.Actual365Fixed())
)

# --- Heston process & model ---
process = ql.HestonProcess(
    rf_curve, div_curve, spot_handle, v0, kappa, theta, sigma, rho
)
model = ql.HestonModel(process)

# --- Analytic (Fourier-based) engine ---
engine = ql.AnalyticHestonEngine(model)

# --- European Option setup ---
payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(calculation_date + int(T * 365))

option = ql.VanillaOption(payoff, exercise)
option.setPricingEngine(engine)

# --- Price ---
call_price = option.NPV()
print(f"Heston (QuantLib) Call: {call_price:.4f}")

# --- Put via parity ---
put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
print(f"Heston Put (by parity): {put_price:.4f}")

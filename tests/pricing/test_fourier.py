from elm.models.pricing.methods.fourier import FourierPricer
from elm.models.pricing.models.heston import HestonModel

hm = HestonModel(
    S0=100, v0=0.04, kappa=2, theta=0.05, sigma=0.3, rho=-0.5, r=0.05, q=0.0
)

fp = FourierPricer(hm)
K, T = 100.0, 1.0
call = fp.price(K, T)  # default is call

put = fp.price(K, T, option_type="put")
print(f"Heston-Fourier Call: {call:.4f}, Put: {put:.4f}")

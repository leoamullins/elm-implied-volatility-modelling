from elm.models.pricing.models.heston import HestonModel
from elm.models.pricing.methods.cos import COSPricer

hm = HestonModel(
    S0=100, v0=0.04, kappa=2, theta=0.05, sigma=0.3, rho=-0.5, r=0.05, q=0.0
)
cos = COSPricer(N=1024, L=15.0)

cf = hm.characteristic_function  # signature (u, T) -> complex

K, T = 100.0, 1.0
call = cos.price(K, T, hm.r, cf, option_type="call")
put = cos.price(K, T, hm.r, cf, option_type="put")
print(f"Heston-COS Call: {call:.4f}, Put: {put:.4f}")

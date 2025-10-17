from elm.models.pricing.methods.monte_carlo import MonteCarlo, print_clean_results
from elm.models.pricing.models.heston import HestonModel


hm = HestonModel(
    S0=100, v0=0.04, kappa=2, theta=0.05, sigma=0.3, rho=-0.5, r=0.05, q=0.0
)

mc_qe = MonteCarlo(hm)
K, T = 100, 1.0
call_mc = mc_qe.price(
    K, T, n_steps=100, n_paths=100000, scheme="qe", seed=42, option_type="call"
)
put_mc = mc_qe.price(
    K, T, n_steps=100, n_paths=100000, scheme="qe", seed=42, option_type="put"
)


print_clean_results(call_mc)
print_clean_results(put_mc)

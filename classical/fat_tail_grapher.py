import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- Model Definitions (from your var_core.py and var_smart.py) ---

@dataclass(frozen=True)
class GaussianModel:
    mu: float
    sigma: float
    def sample_returns(self, rng, n):
        return rng.normal(self.mu, self.sigma, size=n)

@dataclass(frozen=True)
class JumpDiffusionModel:
    mu: float        # Diffusion mean
    sigma: float     # Diffusion volatility
    lamb: float      # Poisson intensity (Expected jumps)
    jump_mu: float   # Mean jump size (negative for market crashes)
    jump_sigma: float # Volatility of the jump size
    
    def sample_returns(self, rng, n):
        # 1. Standard Gaussian Noise
        diffusion = rng.normal(self.mu, self.sigma, size=n)
        # 2. Poisson Jumps
        num_jumps = rng.poisson(self.lamb, size=n)
        jump_impact = np.zeros(n)
        for i in range(n):
            if num_jumps[i] > 0:
                jumps = rng.normal(self.jump_mu, self.jump_sigma, size=num_jumps[i])
                jump_impact[i] = np.sum(jumps)
        return diffusion + jump_impact

# --- Parameters for Visualization ---

# We use a standard 2% daily volatility for the Normal model
mu, sigma = 0.0, 0.02

# We define the Poisson model to have the same base volatility, 
# but add a 1% chance of a massive -10% jump.
lamb = 0.01 
jump_mu = -0.10
jump_sigma = 0.03

gauss_model = GaussianModel(mu=mu, sigma=sigma)
jump_model = JumpDiffusionModel(mu=mu, sigma=sigma, lamb=lamb, jump_mu=jump_mu, jump_sigma=jump_sigma)

# Generate 1,000,000 samples for high-resolution tails
rng = np.random.default_rng(42)
n_samples = 1000000
gauss_rets = gauss_model.sample_returns(rng, n_samples)
jump_rets = jump_model.sample_returns(rng, n_samples)

# Calculate 99% VaR (the 1% worst case)
g_var99 = -np.quantile(gauss_rets, 0.01)
j_var99 = -np.quantile(jump_rets, 0.01)

# --- Plotting ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Subplot 1: Linear Scale
ax1.hist(gauss_rets, bins=500, density=True, alpha=0.5, label='Normal Distribution', color='#3498db')
ax1.hist(jump_rets, bins=500, density=True, alpha=0.5, label='Poisson Jump-Diffusion', color='#e74c3c')
ax1.set_title("Market Distribution: Normal vs. Poisson (Linear)", fontsize=14, fontweight='bold')
ax1.set_xlim(-0.3, 0.1)
ax1.legend()
ax1.grid(alpha=0.3)

# Subplot 2: Log Scale (The "Fat Tail" Evidence)
ax2.hist(gauss_rets, bins=500, density=True, alpha=0.5, label='Normal Distribution', color='#3498db', log=True)
ax2.hist(jump_rets, bins=500, density=True, alpha=0.5, label='Poisson Jump-Diffusion', color='#e74c3c', log=True)

# Highlight VaR lines
ax2.axvline(-g_var99, color='#2980b9', linestyle='--', label=f'Normal 99% VaR: {g_var99:.2%}')
ax2.axvline(-j_var99, color='#c0392b', linestyle='--', label=f'Poisson 99% VaR: {j_var99:.2%}')

ax2.set_title("Log Scale: Visualizing the 'Fat Tail'", fontsize=14, fontweight='bold')
ax2.set_xlabel("Daily Returns", fontsize=12)
ax2.set_ylabel("Probability Density (Log)", fontsize=12)
ax2.set_xlim(-0.4, 0.1)
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()

print(f"Normal 99% VaR: {g_var99:.4f}")
print(f"Poisson 99% VaR: {j_var99:.4f}")
print(f"Difference: {((j_var99/g_var99)-1):.2%} more risk detected by Poisson model.")

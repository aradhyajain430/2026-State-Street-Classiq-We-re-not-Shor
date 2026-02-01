import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass(frozen=True)
class JumpDiffusionModel:
    mu: float; sigma: float; lamb: float; jump_mu: float; jump_sigma: float
    def sample_returns(self, rng, n):
        diffusion = rng.normal(self.mu, self.sigma, size=n)
        num_jumps = rng.poisson(self.lamb, size=n)
        # Optimized for speed
        jump_impact = np.zeros(n)
        jump_indices = np.where(num_jumps > 0)[0]
        for idx in jump_indices:
            j = rng.normal(self.jump_mu, self.jump_sigma, size=num_jumps[idx])
            jump_impact[idx] = np.sum(j)
        return diffusion + jump_impact

def get_tesla_params():
    # Download real TSLA data
    df = yf.download("TSLA", period="2y", progress=False)
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
    
    # Simple fitting: Identifying jumps as events > 3 standard deviations
    mu_tot, sigma_tot = np.mean(returns), np.std(returns)
    is_jump = np.abs(returns - mu_tot) > (3 * sigma_tot)
    jumps = returns[is_jump]
    diffusion = returns[~is_jump]
    
    model = JumpDiffusionModel(
        mu=float(np.mean(diffusion)),
        sigma=float(np.std(diffusion)),
        lamb=float(len(jumps) / len(returns)),
        jump_mu=float(np.mean(jumps) if len(jumps) > 0 else 0),
        jump_sigma=float(np.std(jumps) if len(jumps) > 0 else 0)
    )
    return model, mu_tot, sigma_tot

# 1. Get real Tesla parameters
tsla_model, mu_total, sigma_total = get_tesla_params()

# 2. Generate samples
rng = np.random.default_rng(42)
n_samples = 1000000
tsla_rets = tsla_model.sample_returns(rng, n_samples)
normal_rets = rng.normal(mu_total, sigma_total, size=n_samples)

# 3. Calculate 95% VaR for both models
# Note: VaR is the threshold where only 5% of losses are worse (left tail)
tsla_var_95 = -np.quantile(tsla_rets, 0.05)
gauss_var_95 = -np.quantile(normal_rets, 0.05)

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Subplot 1: Linear Scale
ax1.hist(normal_rets, bins=500, density=True, alpha=0.5, label='Naive Normal Model', color='#3498db')
ax1.hist(tsla_rets, bins=500, density=True, alpha=0.5, label='TSLA Jump-Diffusion', color='#e74c3c')
ax1.axvline(-gauss_var_95, color='#2980b9', linestyle='--', linewidth=2, label=f'Normal 95% VaR: {gauss_var_95:.2%}')
ax1.axvline(-tsla_var_95, color='#c0392b', linestyle='--', linewidth=2, label=f'Poisson 95% VaR: {tsla_var_95:.2%}')
ax1.set_title("TSLA Return Distribution: Linear Scale", fontsize=15, fontweight='bold')
ax1.set_xlim(-0.25, 0.15)
ax1.legend()
ax1.grid(alpha=0.3)

# Subplot 2: Log Scale (Crucial for showing the tail depth)
ax2.hist(normal_rets, bins=1000, density=True, alpha=0.5, label='Naive Normal Model', color='#3498db', log=True)
ax2.hist(tsla_rets, bins=1000, density=True, alpha=0.5, label='TSLA Jump-Diffusion', color='#e74c3c', log=True)

# Add both VaR lines to the log plot
ax2.axvline(-gauss_var_95, color='#2980b9', linestyle='--', linewidth=2, label=f'Normal VaR: {gauss_var_95:.2%}')
ax2.axvline(-tsla_var_95, color='#c0392b', linestyle='--', linewidth=2, label=f'Poisson VaR: {tsla_var_95:.2%}')

# Expand x-axis on the left to show the "Fat Tail" properly (un-zoomed)
ax2.set_xlim(-0.5, 0.5) 
ax2.set_title("TSLA Log-Scale: Visualizing the 'Fat Tail'", fontsize=15, fontweight='bold')
ax2.set_xlabel("Daily Returns (Decimal)")
ax2.set_ylabel("Log Probability Density")
ax2.legend()
ax2.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.show()

print(f"Normal VaR (95%): {gauss_var_95:.5f}")
print(f"Poisson VaR (95%): {tsla_var_95:.5f}")

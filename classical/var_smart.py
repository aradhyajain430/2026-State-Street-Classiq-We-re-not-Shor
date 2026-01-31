import numpy as np
from dataclasses import dataclass
from scipy.stats import poisson, norm

@dataclass(frozen=True)
class JumpDiffusionModel:
    mu: float        # Drift (Diffusion mean)
    sigma: float     # Volatility (Diffusion std)
    lamb: float      # Poisson intensity (Expected jumps per period)
    jump_mu: float   # Mean size of a jump
    jump_sigma: float # Volatility of jump size

    def sample_returns(self, rng: np.random.Generator, n: int) -> np.ndarray:
        # 1. Standard Diffusion component (Gaussian Noise)
        diffusion = rng.normal(self.mu, self.sigma, size=n)
        
        # 2. Poisson component (How many jumps occur in each sample?)
        # For a 1-day VaR, we usually see 0 or 1 jump.
        num_jumps = rng.poisson(self.lamb, size=n)
        
        # 3. Calculate the impact of those jumps
        jump_impact = np.zeros(n)
        for i in range(n):
            if num_jumps[i] > 0:
                # Sum of N jumps, where each jump is Gaussian
                jumps = rng.normal(self.jump_mu, self.jump_sigma, size=num_jumps[i])
                jump_impact[i] = np.sum(jumps)
        
        return diffusion + jump_impact

def calculate_jump_var(model: JumpDiffusionModel, confidence: float, n_samples: int):
    rng = np.random.default_rng()
    returns = model.sample_returns(rng, n_samples)
    losses = -returns # P&L
    return np.quantile(losses, confidence)

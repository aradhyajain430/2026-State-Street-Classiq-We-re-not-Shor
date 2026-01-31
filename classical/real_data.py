# real_data.py
import argparse
import numpy as np
from var_core import GaussianModel, monte_carlo_var, theoretical_var_gaussian
from run_smart import fit_jump_diffusion, calculate_jump_var

def main():
    parser = argparse.ArgumentParser(description="Run VaR analysis with Gaussian or Poisson models.")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Stock ticker (e.g., AAPL, ^GSPC, TSLA)")
    parser.add_argument("--method", type=str, choices=["g", "p", "gp"], default="p", 
                        help="Risk modeling method")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (e.g. 0.95 or 0.99)")
    parser.add_argument("--samples", type=int, default=100000, help="Number of Monte Carlo samples")
    args = parser.parse_args()

    try:
        # We use the smarter fitting function which identifies jumps automatically
        model_params = fit_jump_diffusion(args.ticker)
    except Exception as e:
        print(f"Error: {e}")
        return

    confidence = args.confidence * 100
    if args.method == "g" or args.method == "gp":
        # Standard bell curve logic using the diffusion parameters

        print(f"\n--- Analysis for {args.ticker} (Gaussian Method) ---")
        model = GaussianModel(mu=model_params.mu, sigma=model_params.sigma)
        sim_var = monte_carlo_var(model, args.confidence, args.samples)
        true_var = theoretical_var_gaussian(model, args.confidence)
        
        print(f"Daily Mean (mu):    {model_params.mu:.6f}")
        print(f"Daily Vol (sigma):  {model_params.sigma:.6f}")
        print(f"Theoretical VaR {confidence}%:    {true_var:.6f}")
        print(f"Monte Carlo VaR {confidence}%:    {sim_var:.6f}")

    if args.method == "p" or args.method == "gp":
        # Sophisticated Poisson-stacked logic

        print(f"\n--- Analysis for {args.ticker} (Poisson Method) ---")
        sim_var = calculate_jump_var(model_params, args.confidence, args.samples)
        
        print(f"Poisson Intensity:  {model_params.lamb:.5f} jumps/day")
        print(f"Jump Bias (mu_j):   {model_params.jump_mu:.5f}")
        print(f"{confidence}% VaR:            {sim_var:.6f}")
        print("\nNote: Poisson method accounts for fat tails and crash skewness.")

if __name__ == "__main__":
    main()

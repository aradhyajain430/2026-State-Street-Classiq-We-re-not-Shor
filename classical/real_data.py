import argparse
import yfinance as yf
import numpy as np
from var_core import GaussianModel, monte_carlo_var, theoretical_var_gaussian

def get_real_parameters(ticker, period="1y"):
    """Fetch data and calculate daily log-return stats."""
    print(f"Fetching data for {ticker}...")
    
    # FIX: Set multi_level_index=False to avoid the MultiIndex column error
    df = yf.download(ticker, period=period, progress=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # Calculate Log Returns
    close_prices = df['Close']
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    # Calculate stats and ensure they are raw floats
    # .iloc[0] or .item() safely extracts the number if it's still wrapped in a Series
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    return float(mu), float(sigma)

def main():
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Run VaR analysis on a real stock ticker.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (e.g., AAPL, ^GSPC, TSLA)")
    parser.add_argument("--samples", type=int, default=100000, help="Number of Monte Carlo samples")
    args = parser.parse_args()

    # 2. Get parameters for the specific ticker
    try:
        mu, sigma = get_real_parameters(args.ticker)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 3. Integrate with the core math engine
    model = GaussianModel(mu=mu, sigma=sigma)
    confidence = 0.95
    
    sim_var = monte_carlo_var(model, confidence, args.samples)
    true_var = theoretical_var_gaussian(model, confidence)
    
    # 4. Display Results
    print(f"\n--- Analysis for {args.ticker} ---")
    print(f"Daily Mean (mu):    {mu:.6f}")
    print(f"Daily Vol (sigma):  {sigma:.6f}")
    print(f"Theoretical VaR:    {true_var:.6f}")
    print(f"Monte Carlo VaR:    {sim_var:.6f}")
    print(f"Error:              {abs(sim_var - true_var):.6f}")

if __name__ == "__main__":
    main()

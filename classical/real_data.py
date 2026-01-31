import yfinance as yf
import numpy as np
from var_core import GaussianModel, monte_carlo_var, theoretical_var_gaussian

def get_real_parameters(ticker, period="1y"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period)
    
    # Calculate Log Returns
    close_prices = df['Close']
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    # .mean() and .std() might return a Series if the dataframe is multi-indexed
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    # FIX: Use .item() or .iloc[0] to get the raw float out of the pandas object
    # If mu is a float already, .item() still works; if it's a Series, it grabs the value.
    try:
        final_mu = mu.item()
        final_sigma = sigma.item()
    except AttributeError:
        # If they are already raw numpy floats
        final_mu = float(mu)
        final_sigma = float(sigma)
        
    return final_mu, final_sigma

def main():
    ticker = "TSLA" # Change to any stock (e.g., TSLA, NVDA, SPY)
    mu, sigma = get_real_parameters(ticker)
    
    # Integrate with your teammate's GaussianModel
    stock_model = GaussianModel(mu=mu, sigma=sigma)
    
    # Run a simulation using your teammate's logic
    confidence = 0.95
    n_samples = 100000
    
    sim_var = monte_carlo_var(stock_model, confidence, n_samples)
    true_var = theoretical_var_gaussian(stock_model, confidence)
    
    print(f"\n--- Analysis for {ticker} ---")
    print(f"Daily Mean (mu):    {mu:.6f}")
    print(f"Daily Vol (sigma):  {sigma:.6f}")
    print(f"Theoretical VaR:    {true_var:.6f}")
    print(f"Monte Carlo VaR:    {sim_var:.6f}")
    print(f"Error:              {abs(sim_var - true_var):.6f}")

if __name__ == "__main__":
    main()

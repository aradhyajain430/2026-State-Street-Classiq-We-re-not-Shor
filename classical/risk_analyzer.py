import yfinance as yf
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Import your existing Jump Diffusion logic
from var_smart import JumpDiffusionModel, calculate_jump_var

class RiskAnalyzer:
    def __init__(self):
        """Initializes with no data or model attached."""
        self.model = None
        self.historical_prices = []

    def get_data(self, ticker: str, timeframe_days: int) -> list[float]:
        """
        Fetches historical data and returns a simple list of closing prices.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL').
            timeframe_days: The number of past days to look back.
        """
        # Format the timeframe for yfinance (e.g., '252d')
        period = f"{timeframe_days}d"
        
        # Download data without the extra console progress bars
        df = yf.download(ticker, period=period, progress=False, multi_level_index=False)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}. Check the ticker or timeframe.")

        # Convert the 'Close' column to a standard Python list of floats
        self.historical_prices = df['Close'].dropna().tolist()
        return self.historical_prices

    def fit_and_simulate(self, prices: list[float], confidence: float = 0.95, samples: int = 10000):
        """
        Takes a list of prices, fits the Jump Diffusion model, and returns the VaR.
        """
        if len(prices) < 2:
            raise ValueError("Need at least two price points to calculate returns.")

        # 1. Convert price list to log-returns
        price_array = np.array(prices)
        returns = np.log(price_array[1:] / price_array[:-1])
        
        # 2. Separate routine noise from 'Jumps'
        mu_total, sigma_total = returns.mean(), returns.std()
        threshold = 3.0  # Standard 3-sigma jump detection
        
        is_jump = np.abs(returns - mu_total) > (threshold * sigma_total)
        jumps = returns[is_jump]
        diffusion = returns[~is_jump]
        
        # 3. Build the model object
        self.model = JumpDiffusionModel(
            mu=float(diffusion.mean()),
            sigma=float(diffusion.std()),
            lamb=float(len(jumps) / len(returns)),
            jump_mu=float(jumps.mean() if len(jumps) > 0 else 0),
            jump_sigma=float(jumps.std() if len(jumps) > 0 else 0)
        )
        
        # 4. Run the Monte Carlo simulation
        return calculate_jump_var(self.model, confidence, samples)

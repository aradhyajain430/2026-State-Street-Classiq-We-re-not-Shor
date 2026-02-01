import numpy as np
from scipy.stats import norm
from risk_analyzer import RiskAnalyzer

def run_comparison(ticker, conf):
    print(f"\n--- Backtesting {ticker} ---")
    analyzer = RiskAnalyzer()
    conf = conf

    # 1. Fetch "Past" Data (2024) to train the models
    # We'll use 252 trading days for a full year
    past_prices = analyzer.get_data(ticker, timeframe_days=504) # Get enough to split
    train_prices = past_prices[:252]
    test_prices = past_prices[252:]

    # 2. Calculate Poisson VaR (The "Smart" Model)
    poisson_var = analyzer.fit_and_simulate(train_prices, confidence=conf)

    # 3. Calculate Gaussian VaR (The "Classical" Model)
    train_rets = np.log(np.array(train_prices[1:]) / np.array(train_prices[:-1]))
    mu_g, sigma_g = train_rets.mean(), train_rets.std()
    gaussian_var = -(mu_g + sigma_g * norm.ppf(1 - conf))

    # 4. Run the "Future" Test (The next year)
    test_rets = np.log(np.array(test_prices[1:]) / np.array(test_prices[:-1]))
    actual_losses = -test_rets
    
    p_breaches = np.sum(actual_losses > poisson_var)
    g_breaches = np.sum(actual_losses > gaussian_var)
    expected = len(actual_losses) * (1 - conf)


    print(f"Poisson VaR Predicts: {poisson_var:.2%}")
    print(f"Gaussian VaR Predicts: {gaussian_var:.2%}")
    print("-" * 30)
    print(f"Results over {len(actual_losses)} days:")
    print(f"Poisson Breaches:  {p_breaches} (Accuracy: {1 - abs(p_breaches-expected)/expected:.1%})")
    print(f"Gaussian Breaches: {g_breaches} (Accuracy: {1 - abs(g_breaches-expected)/expected:.1%})")
    print(f"Expected Breaches: {expected:.1f}")

if __name__ == "__main__":
    # Test on a stable index and a volatile stock
 

    print("KEEPERS")
    print("S&P")
    run_comparison("SPY", 0.95)
    print("\nTesla")
    run_comparison("TSLA", 0.95)
    print("\nNvidia")
    run_comparison("NVDA", 0.95)
    print("\nCoinbase")
    run_comparison("COIN", 0.95)
    print("\nMongoDB")
    run_comparison("MDB", 0.95)
    print("\nRivian")
    run_comparison("RIVN", 0.95)

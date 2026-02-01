import matplotlib.pyplot as plt
import numpy as np

# Data from the table
tickers = ['SPY', 'TSLA', 'NVDA', 'COIN', 'MDB', 'RIVN']
poisson_accuracy = [48.6, 96.4, 79.7, 71.7, 80.5, 39.8]
gaussian_accuracy = [48.6, 71.7, 63.7, 23.9, 55.8, 0.0]

x = np.arange(len(tickers))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create the bars
rects1 = ax.bar(x - width/2, poisson_accuracy, width, label='Poisson', color='#3498db')
rects2 = ax.bar(x + width/2, gaussian_accuracy, width, label='Gaussian', color='#e74c3c')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('VaR Backtesting Accuracy: Poisson vs. Gaussian')
ax.set_xticks(x)
ax.set_xticklabels(tickers)
ax.legend()

# Adding data labels on top of bars
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

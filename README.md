
# We're not Shor Final Submission
Team members: Eashan Iyer, (write your names)
Brown University

Presentation: https://docs.google.com/presentation/d/14nTP47RFCd-AP16AxCKQe18zIftDFV3tF5MNip__Ehk/edit?slide=id.g1ed75f95cf8_1_28099#slide=id.g1ed75f95cf8_1_28099 

## The VaR problem:
VaR or value at risk is a quantity that measures the maximum expected loss 
over a given time period at a certain confidence level. For instance, 
if your VaR for a one day time frame at a 95% confidence interval is 
1,000,000 dollars, then you would lose more than 1,000,000 dollars 
in a single day due to random variation only 5% of the time. 

## Classical Solution
We were able to conceptualize the classical solution very quickly and many built
in packages already contained the code we needed to implement this solution.

Given a normal Guassian distribution function with variations within ne day
and confidence level c, we can solve for VaR with the following:
Assuming daily returns are normally distributed,
\[
R \sim \mathcal N(\mu, \sigma),
\]
the Value at Risk (VaR) at confidence level \(c\) is computed using the
lower-tail quantile of the distribution.

```python
from statistics import NormalDist

z = NormalDist().inv_cdf(1 - c)
VaR = portfolio_value * (mu - sigma * z)
```
Where the distribution function measures the percent change in the portfolio 
over a single day.

Essentially, we find when the cdf equals 1-c and get the z-score at that point. 
We then use that z-score calculate the percent change which is equal to 
(mu - sigma * z). Then we multiply the percent change by the portfolio value to 
get the net loss in portfolio value.


## Quantum Solution


# Summary 

Your final writeup should include:  

A clear description of the VaR problem and assumptions behind your probability model.
An explanation of your classical Monte Carlo workflow.
A description of your quantum AE/IQAE workflow and bisection search.
Plots comparing accuracy vs number of probability queries.
Sensitivity analysis covering discretization, precision, and confidence levels.
A discussion of:
when quantum methods appear advantageous,
what assumptions enable the advantage,
what aspects are asymptotic vs simulator artifacts.
Creativity is encouraged. Tell a story with your methodology, results, and insights.


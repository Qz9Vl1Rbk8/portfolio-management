import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from efficient_frontier import *


# Plot the efficient frontier
if __name__ == '__main__':

    # Initialize the figure
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    # Define tickers to load
    tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'CSCO',
               'ORCL', 'AMD', 'QCOM', 'BLK', 'JPM', 'NFLX', 'TSLA']

    # Load the data
    data = {tick: pd.read_csv(f'data/{tick}.csv') for tick in tickers}

    # Extract the adjusted close prices
    close = np.array(
        [data[tick]['adjClose'].values.tolist()[::-1]
            for tick in tickers]
    ).T

    # Calculate the rate of return
    rate_of_return = compute_rate_of_return(close)

    # Get and plot the Gaussian parameters w.r.t. rate of return
    stddev, mean, covar = get_gaussian_params(rate_of_return)
    for tick, stddev_i, mean_i in zip(tickers, stddev, mean):
        ax.scatter(stddev_i, mean_i, color='blue', s=7)
        ax.annotate(tick, xy=(stddev_i, mean_i))

    # Optimize the portfolio
    # target_return = 0.0005
    # weights = opt_portfolio(covar, mean, target_return)
    # print(weights)

    # Use the covariance matrix and the mean to compute the efficient frontier
    ex, ey = comp_efficient_frontier(covar, mean)
    ax.plot(ex, ey, color='red', linewidth=0.8)

    # Find min. variance portfolio
    _, mx, my = opt_portfolio(covar, mean, None)
    ax.scatter(mx, my, color='green', s=45)

    # Finalize the plot
    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Average Daily Return')
    ax.set_title('Effecient Frontier Chart')
    plt.show()

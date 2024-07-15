import numpy as np


# Compute the rate of return
def compute_rate_of_return(prices):
    return prices[1:] / prices[:-1] - 1.0


# Get the Gaussian parameters (standard deviation, mean, covariance)
def get_gaussian_params(samples):
    mean = np.mean(samples, axis=0)
    covar = np.cov(samples, rowvar=False)
    stddev = np.sqrt(np.diag(covar))
    return stddev, mean, covar


# Optimize the portfolio
def opt_portfolio(cov_matrix,
                  expected_returns,
                  target_return):
    
    # Measure the number of assets
    num_assets = len(cov_matrix)

    # Check if the target return is provided
    # If not, just minimize the variance
    if target_return:
        # Augment the covariance matrix to include the constraints
        cov_mod = np.zeros((num_assets + 2, num_assets + 2))
        cov_mod[:num_assets, :num_assets] = cov_matrix
        cov_mod[:num_assets, num_assets] = expected_returns
        cov_mod[num_assets, :num_assets] = expected_returns
        cov_mod[:num_assets, num_assets + 1] = 1.0
        cov_mod[num_assets + 1, :num_assets] = 1.0

        # Set the last two rows/columns of the augmented matrix
        cov_mod[num_assets, num_assets] = 0.0
        cov_mod[num_assets, num_assets + 1] = target_return
        cov_mod[num_assets + 1, num_assets] = target_return
        cov_mod[num_assets + 1, num_assets + 1] = 1.0

        # Construct the constraints vector
        constraints_vector = np.zeros(num_assets + 2)
        constraints_vector[num_assets] = target_return
        constraints_vector[num_assets + 1] = 1.0

    # Just minimize the variance
    else:
        # Augment the covariance matrix to include the constraints
        cov_mod = np.zeros((num_assets + 1, num_assets + 1))
        cov_mod[:num_assets, :num_assets] = cov_matrix
        cov_mod[:num_assets, num_assets] = 1.0
        cov_mod[num_assets, :num_assets] = 1.0
        cov_mod[num_assets, num_assets] = 1.0

        # Construct the constraints vector
        constraints_vector = np.zeros(num_assets + 1)
        constraints_vector[num_assets] = 1.0

    # Solve the linear system to find the portfolio weights
    weights_mod = np.linalg.solve(cov_mod, constraints_vector)
    weights = np.array(weights_mod[:num_assets])

    # Compute the portfolio variance and return
    result_var = np.sqrt(weights.T.dot(cov_matrix.dot(weights)))
    result_return = weights.T.dot(expected_returns)

    return weights, result_var, result_return


# Compute the efficient frontier function
def comp_efficient_frontier(covar, mean, return_step=0.0001):
    result_vars, result_returns = [], []
    for target_return in np.arange(np.min(mean), np.max(mean) + return_step, return_step):
        # Get the portfolio weights
        weights, result_var, result_return = opt_portfolio(
            covar, mean, target_return
        )

        # Compute the corresponding risk and return
        result_vars.append(result_var)
        result_returns.append(result_return)

    # Return the efficient frontier points
    return result_vars, result_returns

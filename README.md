### Portfolio Optimization Using Matrix Inversion
In portfolio optimization, the goal is to construct a portfolio that minimizes risk while achieving a target return.
This involves solving a system of linear equations derived from the optimization problem.

The objective: Minimum risk portfolio with a target return. This means,

$\min_{w} w^T \Sigma w$

where $w$ is the weight vector of the portfolio and $\Sigma$ is the covariance matrix of the assets in the portfolio. Furthermore, the portfolio should follow two constraints:

1. The sum of the weights should be 1: $\sum_{i=1}^{n} w_i = 1$

2. The expected return of the portfolio should be equal to a target return: $w^T \mu = r$

where $\mu$ is the expected return vector of the assets in the portfolio and $r$ is the target return.

To find the optimal portfolio, we can solve the system of linear equations:

$A w = b$ where $A = \begin{bmatrix} \Sigma & \mu & 1 \\ \mu^T & 0 & 0 \\ 1 & 0 & 0 \end{bmatrix}$ and $b = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ r \\ 1 \end{bmatrix}$.

Given a full rank, not singular and square matrix $A$, we can solve the system of linear equations using matrix inversion, $w = A^{-1} b$.
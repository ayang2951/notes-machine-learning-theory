_Overview: Complexity terms of classes with a decreasing number of assumptions._

## Model Classes and Complexity

Let us set up the discussion on complexity. We fix a model class $\mathcal F = \\{f : \mathcal X \rightarrow \mathcal Y\\}$, and for some loss function $\ell(f(X), Y)$ we define risk $R(f) = \mathbb E_{X, Y}[\ell(f(X), Y)]$. For a sample $S = \\{(X_i, Y_i)\\}\_{i = 1}^n$, we have empirical risk 
$$\hat R(f) = \frac{1}{n} \sum_{i = 1}^n \ell(f(X_i), Y_i),$$
and hence
$$\mathbb E_{S}[\hat R(f)] = \frac{1}{n} \sum_{i = 1}^n \mathbb E_{(X_i, Y_i)} \ell(f(X_i), Y_i) = R(f).$$
Again, in the empirical risk minimization framework, we have
$$\hat f = \arg\min_{f \in \mathcal F}\hat R(f).$$
Now, we will show the connection between 
<ol type="i">
  <li>the difference in empirical risk and the true (theoretical quantity) risk, and</li>
  <li>the excess risk,</li>
</ol>
where for the latter concept we have already established the relationship with the complexity $\mathcal C(\mathcal F)$.



Suppose that $f^** \in \mathcal F$ and hence that $f^* = f^**$. We now describe one way in which we can bound the excess risk. Suppose that we have <i>uniform convergence of the difference between the risk and empirical risk</i>, i.e.
$$\sup_{f \in \mathcal F}|R(f) - \hat R(f)| \leq \epsilon.$$
We would therefore have that
$$R(\hat f) \overset{(1)}\leq \hat R(\hat f) + \epsilon \overset{(2)}\leq \hat R(f^\*) + \epsilon \overset{(3)}\leq (R(f^\*) + \epsilon) + \epsilon,$$
where $(1)$ and $(3)$ are both by uniform convergence of empirical risk and $(2)$ is because $\hat f$ is the empirical risk minimizer. Hence, we conclude that
$$R(\hat f) - R(f) \leq 2 \cdot \epsilon.$$
Thus, if we are able to bound the sup difference in risk over the function class, we get a bound on the excess risk. We now discuss increasingly general cases for the model class.

### Model Complexity for Finite Model Classes


_Overview: Introduction to learning, learners, risk, and empirical risk minimization._

## Learners, Risk, and Estimators

Our focus will be on regression and classification tasks in which we have features (predictors), $X \in \mathbb R^d$, and an outcome, $Y \in \mathbb R$.

In the setup of our learning problem, we have independently and identically distributed data, called a <i>sample</i>: 
$$S \triangleq \\{(X_i, Y_i)\\}\_{i = 1}^n \sim \mathbb P\_{X, Y},$$
and our objective is to produce a <i>learner</i> based on the sample and return a function 
$$\hat f = \hat f\_S(X),$$
which is a function (of <i>only the features $X$</i>) of some future input $(X, Y) \sim \mathbb P\_{X, Y}$, and may be used to predict the outcome of future data when we are given only the features.



### Defining the Risk

We are typically interested in the performance of our fitted model on unseen, future data, and we shall formalize this using the <i>risk</i>. Concretely, to discuss the risk, we need to first define a <i>loss function</i>.



<div class="callout definition"><span class="label">Definition: Loss Function</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For an estimation problem, given a function of the data $f(X)$ and the target (outcome) $Y$, a <strong><i>loss function</strong></i> is a numerical function
$\ell(f(X), Y)$ where
<ol type="i">
  <li>$\ell(f(X), Y) \geq 0$ for all $f$ and $Y$.</li>
  <li>If $f(X) = Y$, $\ell(f(X), Y) = 0$.</li>
</ol>
</div>



We now provide a few examples of loss functions that are commonly used.



<div class="callout example"><span class="label">Example: Common Loss Functions</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
First, consider a regression problem where $Y \in \mathbb R$ is continuous. A common loss function to use is $L_2$ loss, 
$$\ell(f(X), Y) = (f(X) - Y)^2.$$

Next, consider a classification problem where the outcome $Y \in \\{0, 1\\}$ is binary. A common loss function to use is zero-one, 
$$\ell(f(X), Y) = \mathbb I_{\\{f(X) \neq Y\\}}.$$
There are many other options for both regression and classification tasks.
</div>


Now, the risk may be defined for an arbitrary loss function.



<div class="callout definition"><span class="label">Definition: Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For an arbitrary loss function $\ell(f(X), Y)$, the <strong><i>risk</i></strong> of the loss function is defined as
$$R(f, Y) \triangleq \mathbb E_{X}[\ell(f(X), Y)].$$
</div>



For a given loss function and estimator $\hat f$, the randomness in the new random variable is "integrated out". Note that $\hat f$ is still a random quantity, whose randomness comes from the sample. 



<div class="callout remark"><span class="label">Remark: Risk of $f(X)$</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
We are currently not discussing the estimation of a regression function $f$, but rather discussing how $Y$ is estimated from $x$ with a regression function $f$.
</div>



In both regression and classification problems, the universe of models is too large, and we will choose a class $\mathcal F$ to focus our attention on (***justify). 

*** add example maybe For example, in linear regression with $d$ features, we may define $\mathcal F = \\{f_w(x) = w^T x : w \in \mathbb R^d\\}$.

Once we do this, two main questions emerge. Let us consider our chosen class $\mathcal F$ to be the <i>comparator class</i> against which we compare our fitted model. Again, choosing to restrict our attention to the class $\mathcal F$ , the following questions are what we are interested in asking:
<ol type="1">
<li>First, we ask whether the risk of the fitted model $\hat f$ converges to the infimum risk over the class, i.e. whether
$$R(\hat f) - \inf_{f \in \mathcal F}R(f) \rightarrow 0$$
as $n \rightarrow \infty$. 
</li>
<li>The second question is, <i>if</i> the risk converges to the inf risk over the comparator class, at what rate?</li>
</ol>



### Convergence of Risk

Still focusing on a chosen comparator class $\mathcal F$, we compare the risk of a fitted model only to the infimum over that particular class. We define the following quantity to make this precise.



<div class="callout definition"><span class="label">Definition: Excess Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $\mathcal F$ be a comparator class, and let $\hat f$ be a fitted model on a sample $S$. The <strong><i>excess risk</i></strong> of the model $\hat f$ is defined:
$$\mathcal E(\hat f) \triangleq R(\hat f) - \inf_{f \in \mathcal F}R(f).$$
</div>



Now, having defined the excess risk, we may define types of convergence of the excess risk that we are interested in.



<div class="callout definition"><span class="label">Definition: Convergence in Expectation of Excess Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and an estimator $\hat f_n$ of the outcome $Y$ (denoted thus to make clear that it is dependent on the sample and its size), the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in expectation</i></strong> if, for a sequence $\alpha(n, \mathcal F) = o(1)$,
$$\mathbb E_{S}[\mathcal E(\hat f_n)] \leq \alpha (n, \mathcal F).$$
</div>



Notationally, $\alpha(n, \mathcal F)$ is a sequence dependent on the size of the sample <i>and</i> the chosen comparator class $\mathcal F$. For example, in linear regression in dimension $d$ and thus defined $\mathcal F = \\{f_w(x) = w^T x : w \in \mathbb R^d\\}$, we have that $\alpha(n, \mathcal F) = \frac{d}{n}$.

The second type of convergence we may care about is convergence in probability.



<div class="callout definition"><span class="label">Definition: Convergence in Probability of Excess Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and a estimator $\hat f_n$, the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in probability</i></strong> if
$$\mathbb P_{S}(\mathcal E(\hat f_n) \leq \alpha (n, \mathcal F, \delta)) \geq 1 - \delta.$$
</div>



Statisticians may primarily be interested in the first kind of convergence, as it is stronger: convergence in $L_1 \implies$ convergence in probability. However, a particular case of import for the machine learning community is when $\alpha(n, \mathcal F, \delta)$ has a particular rate, which we define below.



<div class="callout definition"><span class="label">Definition: Convergence in Probability with High Probability</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and an estimator $\hat f_n$, the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in probability with high probability</i></strong> if
$$\mathbb P_{S}(\mathcal E(\hat f_n) \leq \alpha (n, \mathcal F, \delta)) \geq 1 - \delta$$
where $\alpha(n, \mathcal F, \delta)$ is on the order of $\log \frac{1}{\delta}$.
</div>



For example, if $\mathcal E(\hat f_n) \leq \alpha(n, \mathcal F, \delta) = -\frac{d \log \delta}{n}$ with probability $1 - \delta$.

This particular convergence is desirable, as it actually can be upgraded to convergence almost surely of the excess risk, as stated by the proposition below.



<div class="callout proposition"><span class="label">Proposition: High Probability Convergence Implies Almost Sure Convergence</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and an estimator $\hat f_n$, suppose that the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in probability with high probability</i></strong>. Then the excess risk converges to 0 almost surely.
</div>



<details class="collapsible">
  <summary>Proof</summary>
  <div class="collapsible__content">
  proof.
  </div>
</details>



### Types of Estimators

There are many types of estimators that we may choose. One natural choice is <i>empirical risk minimizer</i> estimators, which is related to maximum likelihood estimators and M-estimators (maximum-likelihood type, wherein we generalize a maximization over the likelihood to some convex function).

We first define the empirical risk. As with other "empirical" objects, we just "plug in" the data where we have unknown quantities. We are aiming to use $f(X)$ to estimate $Y$, so we just plug in $Y$.



<div class="callout definition"><span class="label">Definition: Empirical Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a sample $\\{(X_i, Y_i)\\}_{i = 1}^n$, a function on the predictors $f(X)$, the outcome $Y$, and a loss function $\ell(f(X), Y)$, the <strong><i>empirical risk</i></strong> of $f$ is defined as
$$\hat R(f) = \frac{1}{n} \sum_{i = 1}^n \ell(f(X_i), Y_i).$$
</div>



At this point, if we have fixed the class $\mathcal F$, for instance taking it to be the set of all linear classifiers (for classification, obviously), or the set of all neural networks, we may aim to minimize the empirical risk within this class. Hence, we can obtain the empirical risk minimizer.



<div class="callout definition"><span class="label">Definition: Empirical Risk Minimizer</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a class $\mathcal F$ and a sample $\\{(X_i, Y_i)\\}_{i = 1}^n$, the <strong><i>empirical risk minimizer</i></strong> is the function $f$ in the class $\mathcal F$ that minimizes the empirical risk:
$$\hat f = \arg\min_{f \in \mathcal F} \hat R(f).$$
</div>



Another type of estimator is a plug-in estimator, which tries to estimate the <i>true</i> minimizer of risk, which we call $\hat f$, over a universal class of all models, $\mathcal U$. We may define this $\hat f$ once we have chosen a loss and hence the risk $R$:
$$f^{**} \triangleq \arg\min_{f \in \mathcal U}R(f).$$
We note that this is a theoretical quantity.



<div class="callout example"><span class="label">Example: Linear Regression with $L_2$ Loss</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
We are using square error loss, $\ell(f(X), Y) = (f(X) - Y)^2$. Suppose that $Y \in L_2$. Then the function
$$f^{**}(X) = \mathbb E[Y | X = x]$$ minimizes $R(f)$.
</div>



<details class="collapsible">
  <summary>Proof</summary>
  <div class="collapsible__content">
  Let us define $f^{**}(X) \triangleq \mathcal E[Y | X = x]$. We then have, for an arbitrary $f$, the risk $R(f) = \mathbb E[(f(X) - Y)^2]$:
  $$\mathbb E[(f(X) - f^{**}(X) + f^{**}(X) - Y)^2] = \mathbb E[(f(X) - f^{**}(X))^2] + \mathbb E[(f^{**}(X) - Y)^2] + 2 \cdot \mathbb E[(f(X) - f^{**}(X)) \cdot (f^{**}(X) - Y)].$$
  Simplifying the last term in the sum of expectations using the tower property and explicitly writing out $f^{**}(X)$, we have
  $$\mathbb E[\mathbb E[(f(X) - \mathbb E[Y | X]) \cdot (\mathbb E[Y | X] - Y)] | X] = \mathbb E[(f(X) - \mathbb E[Y | X]) \cdot (\mathbb E[Y | X] - \mathbb E[Y | X])] = 0,$$
  where the first equality is by pulling out the first factor, which is known when conditioning on $X$.
  Hence, we have that
  $$R(f(X), Y) = R(f^{**}(X), Y) + \mathbb E[(f(X) - f^{**}(X))^2],$$
  and we conclude that $R(f^{**}) \leq R(f)$ for any arbitrary $f$, and $f^{**}$ is the minimizer of $L_2$ loss.
  </div>
</details>



Since $f^{**}(x) = \mathbb E[Y | X = x]$ is a theoretical quantity, it must be estimated. The question of how to estimate it now requires attention. 

Once choice for a regression problem that would be natural in many settings is to take some weighted average of the points in a neighborhood $B(x)$ for any given $x$ in the domain. For example, this could look like
$$B(x) = B(x, r) \cap \\{X_i\\}_{i = 1}^n.$$
One example of this is $k$-nearest neighbors (KNN), wherein the weighting is 1 for all $k$ neighbors of a value $x$ and 0 for all other datapoints.

Analogously, suppose that we have a binary classification problem where $Y \in \\{0, 1\\}$. A common loss function to use is the zero-one loss $\ell(f(X), Y) = \mathbb I_{\\{f(X) \neq Y\\}}$, and hence we have risk $R(f(X), Y) = \mathbb E[ \ \mathbb I_{\\{f(X) \neq Y\\}} \ ]$.



<div class="callout example"><span class="label">Example: Binary Classification with Zero-One Loss</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
We are using the zero-one loss $\ell(f(X), Y) = \mathbb I_{f(X) \neq Y}$. Then the classifier
$$h^{**}(X) \triangleq \text{sign} (f^{**}(X)) = \text{sign} (\mathbb E[Y | X])$$ minimizes the risk $R(h)$. This is called the <i>Bayes optimal classifier</i>.
</div>



<details class="collapsible">
  <summary>Proof</summary>
  <div class="collapsible__content">
  proof.
  </div>
</details>



An estimator of the Bayes optimal classifier can be analogous to the above for the regression task: we do (possibly weighted) majority voting for the datapoints in the sample within a neighborhood of any given $x$. So, for instance, we can have an estimator of $h^{**}(x)$ be $\hat h\_B(x) = \text{sign}(\hat f\_B(x))$, which is the majority vote within the $B$ neighborhood of any $x$ in the domain.



<div class="callout remark"><span class="label">Remark: Optimal in the Universe vs. in a Class</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
One nuance to be made clear about the function $f^{**}$ is that it is the minimizer over the universe $\mathcal U$. We often restrict the class over which we optimize (or compare) to a class $\mathcal F$ for some type of practicality.

If we are discussing the optimizer in that particular class, we have
$$f^{*} = \arg \min_{f \in \mathcal F} R(f).$$

Sometimes, it is prudent to simply assume that $f^* = \arg\min_{f \in \mathcal F}R(f) \in \mathcal F$, where $\mathcal F$ is some appropriate class, for example the class of all Lipschitz functions, or Sobolev or Hölder.
</div>



## Convergence Rates

We are typically interested in the convergence and convergence rates of the <i>excess</i> risk rather than the risk itself. This is because we generally cannot expect the convergence of the actual risk: for example, in the case of regression, the noise of a new datapoint results in an irreducible term in the risk. Hence, we instead look at the convergence (and rates) for the excess risk.

In the literature&mdash;for instance, for ERM (empirical risk minimization)&mdash;we may see convergence rates for the excess risk of the form
$$\mathcal E(\hat f) = \left(\frac{\mathcal C(\mathcal F)}{n}\right)^{\alpha},$$
where $\mathcal C(\mathcal F)$ is a <i>complexity term</i> for the class $\mathcal F$.



<div class="callout remark"><span class="label">Remark: Complexity Term</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
The complexity term $\mathcal C(\mathcal F)$ is a complex object, and how precisely it should and will be defined depends on the problem. We will build this definition over time and see its relation to concepts such as VC dimension, covering numbers, entropy numbers, Rademacher complexity, etc.
</div>



There are two quantities that help motivate our understanding of complexity. 
<ol type="1">
<li>First, we may think about the variance of the estimator, which is computed from the sample $S$: 
$$\text{Var}_{S}(\hat f).$$</li>
<li>Second, we may be interested in the "variability" of the empirical risk across different samples: for instance, for two equally-sized samples $S_1$ and $S_2$, we might be interested in the quantity
$$|\hat R_{S_1}(\hat f) - \hat R_{S}(\hat f)|.$$</li>
</ol>



To related quantities for the latter are
$$\hat R(\hat f) - \mathbb E[\hat R(\hat f)] \qquad \text{and}  \qquad \hat R(\hat f) - R(\hat f).$$
These quantities seek to quantify the "variability" of the empirical risk of the estimator.



### Complexity of Classes in Regression

We now make the notion more concrete by looking at regression. In linear and polynomial regression, we have a number of predictors $X \in \mathbb R^d$, and an outcome $Y \in \mathbb R$. We may consider polynomials of degree $p$, and to do this, introduce the following notation: for $x \in \mathbb R^d$, $\alpha \in \mathbb N^d$, let $|\alpha| \triangleq \sum\_{i = 1}^d \alpha i$ and $x^\alpha \triangleq \prod\_{i = 1}^d x_{i}^{\alpha_i}$. Hence, any polynomial of degree $p$ may be written as

$$f(x) = \sum\_{|\alpha| \leq p} a_{\alpha} x^\alpha.$$
In this setting, we may view the complexity as degrees of freedom. The larger the specified degree $p$ for the polynomial, the larger the variability of $\hat f$ over different samples. 







<div class="callout example"><span class="label">Example: Polynomial Regression with $p \geq n$</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Consider the extreme case where the degree $p$ is larger than or equal to the number of datapoints $n$. 

Then (provided the $X$ values are all distinct&mdash;which would be the case if $X$ were drawn from a continuous distribution) the curve perfectly fits all $n$ datapoints for any sample of size $n$, resulting in empirical risk of 0.

<p style="color: red;">[include diagram].</p>

<p align="center">
  <img src="../images/picture.jpg" alt="Fitted Polynomials" width="800"/>
</p>

</div>
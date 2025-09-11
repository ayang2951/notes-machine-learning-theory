_Overview: Introduction to learning, learners, risk, and empirical risk minimization._

## Section 1.1: Learners, Risk, and Estimators

Our focus will be on regression and classification tasks in which we have features (predictors), $X \in \mathbb R^d$, and an outcome, $Y \in \mathbb R$.

In the setup of our learning problem, we have independently and identically distributed data, called a <i>sample</i>: 
$$S \triangleq \\{(X_i, Y_i)\\}\_{i = 1}^n \sim \mathbb P\_{X, Y},$$
and our objective is to produce a <i>learner</i> based on the sample and return a function 
$$\hat f = \hat f\_S(X),$$
which is a function of <i>only the features $X$</i> of a future input $(X, Y) \sim \mathbb P\_{X, Y}$, and may be used to predict the outcome of future data when we are given only the features.



### Section 1.1.1: Defining the Risk

We are typically interested in the performance of our learner on unseen, future data, and we shall formalize this using the <i>risk</i>. Concretely, to discuss the risk, we need to first define a <i>loss function</i>.



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
First, consider a regression problem where $Y \in \mathbb R$ is continuous. For a model $f(X)$, a common loss function to use is $L_2$ loss, 
$$\ell(f(X), Y) = (f(X) - Y)^2.$$

Next, consider a classification problem where the outcome $Y \in \\{0, 1\\}$ is binary. A common loss function to use is zero-one, 
$$\ell(f(X), Y) = \mathbb I_{\\{f(X) \neq Y\\}}.$$
There are many other options for both regression and classification tasks.
</div>


Now, the risk may be defined for an arbitrary loss function.



<div class="callout definition"><span class="label">Definition: Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For an arbitrary loss function with a model $f$, $\ell(f(X), Y)$, the <strong><i>risk</i></strong> of the loss function is defined as
$$\mathbb E_{X}[\ell(f(X), Y)].$$
</div>



For a given loss function and estimator $\hat f$, the randomness in the new random variable is "integrated out". Note that $\hat f$ is still a random quantity, whose randomness comes from the sample. 



<div class="callout remark"><span class="label">Remark: Risk of a Model</span><br/>
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



### Section 1.1.2: Convergence of Risk

Still focusing on a chosen comparator class $\mathcal F$, we compare the risk of a fitted model only to the infimum over that particular class. We define the following quantity to make this precise.



<div class="callout definition"><span class="label">Definition: Excess Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $\mathcal F$ be a comparator class, and let $\hat f$ be a fitted model on a sample $S$. The <strong><i>excess risk</i></strong> of the model $\hat f$ is defined:
$$\mathcal E(\hat f) \triangleq R(\hat f) - \inf_{f \in \mathcal F}R(f).$$
</div>



Now, having defined the excess risk, we may define types of convergence of the excess risk that we are interested in.



<div class="callout definition"><span class="label">Definition: Convergence in Expectation of Excess Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and a model $\hat f_n$ (denoted thus to make clear that it is dependent on the sample and its size), the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in expectation</i></strong> if, for a sequence $\alpha(n, \mathcal F) = o(1)$,
$$\mathbb E_{S}[\mathcal E(\hat f_n)] \leq \alpha (n, \mathcal F).$$
</div>



Notationally, $\alpha(n, \mathcal F)$ is a sequence dependent on the size of the sample <i>and</i> the chosen comparator class $\mathcal F$. For example, in linear regression in dimension $d$ and thus defined $\mathcal F = \\{f_w(x) = w^T x : w \in \mathbb R^d\\}$, we have that $\alpha(n, \mathcal F) = \frac{d}{n}$.

The second type of convergence we may care about is convergence in probability.



<div class="callout definition"><span class="label">Definition: Convergence in Probability of Excess Risk</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and a model $\hat f_n$, the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in probability</i></strong> if
$$\mathbb P_{S}(\mathcal E(\hat f_n) \leq \alpha (n, \mathcal F, \delta)) \geq 1 - \delta.$$
</div>



Statisticians may primarily be interested in the first kind of convergence, as it is stronger: convergence in $L_1 \implies$ convergence in probability. However, a particular case of import for the machine learning community is when $\alpha(n, \mathcal F, \delta)$ has a particular rate, which we define below.



<div class="callout definition"><span class="label">Definition: Convergence in Probability with High Probability</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and a model $\hat f_n$, the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in probability with high probability</i></strong> if
$$\mathbb P_{S}(\mathcal E(\hat f_n) \leq \alpha (n, \mathcal F, \delta)) \geq 1 - \delta$$
where $\alpha(n, \mathcal F, \delta)$ is on the order of $\log \frac{1}{\delta}$.
</div>



For example, if $\mathcal E(\hat f_n) \leq \alpha(n, \mathcal F, \delta) = -\frac{d \log \delta}{n}$ with probability $1 - \delta$.

This particular convergence is desirable, as it actually can be upgraded to convergence almost surely of the excess risk, as stated by the proposition below.



<div class="callout proposition"><span class="label">Proposition: High Probability Convergence Implies Almost Sure Convergence</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For a comparator class $\mathcal F$ and a model $\hat f_n$, suppose that the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in probability with high probability</i></strong>. Then the excess risk converges to 0 almost surely.
</div>



<details class="collapsible">
  <summary>Proof</summary>
  <div class="collapsible__content">
  proof.
  </div>
</details>



### Section 1.1.3: Types of Estimators

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



Another type of estimator is a plug-in estimator, which tries to estimate the true minimizer of risk over a universal class of all models, $\mathcal U$. Once we have chosen a loss and hence the risk $R$, we have:
$$f^* = \arg\min_{f \in \mathcal U}R(f).$$
We note that this is a theoretical quantity.



<div class="callout example"><span class="label">Example: Linear with $L_2$ Loss</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let us use square error loss, $\ell(f(X), Y) = (f(X) - Y)^2$. Suppose that $Y \in L_2$. Then the function
$$f^{**}(X) = \mathcal E[Y | X = x]$$ minimizes $R(f)$.
</div>



<details class="collapsible">
  <summary>Proof</summary>
  <div class="collapsible__content">
  Let us define $f^{**}(X) \triangleq \mathcal E[Y | X = x]$. We then have, for an arbitrary $f$, 
  $$R(f) = $$
  </div>
</details>

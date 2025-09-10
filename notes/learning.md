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



In both regression and classification problems, the universe of models is too large, and we will choose a class $\mathcal F$ to focus our attention on (***justify). For example, in linear regression with $d$ features, we may define $\mathcal F = \\{f_w(x) = w^T x : w \in \mathbb R^d\\}$.

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
For a comparator class $\mathcal F$ and a model $\hat f_n$ (denoted thus to make clear that it is dependent on the size of the sample), the excess risk $\mathcal E(\hat f_n)$ <strong><i>converges in expectation</i></strong> if, for a sequence $\alpha(n, \mathcal F) = o(1)$,
$$\mathbb E_{S}[\mathcal E(\hat f_n)] \leq \alpha (n, \mathcal F).$$
</div>



Notationally, $\alpha(n, \mathcal F)$ is a sequence dependent on the size of the sample <i>and</i> the chosen comparator class $\mathcal F$. For example, in linear regression in dimension $d$ and thus defined $\mathcal F = \\{f_w(x) = w^T x : w \in \mathbb R^d\\}$, we have that $\alpha(n, \mathcal F) = \frac{d}{n}$.

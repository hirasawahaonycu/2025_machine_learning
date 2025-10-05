# Week 5 Assignment


## Problem 1

Given

$$
f(x) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

where $x,\mu \in \mathbb{R}^k$ , $\Sigma$ is a $k$ -by- $k$ positive definite matrix and $|\Sigma|$ is its determinant

Denote $\Sigma^{\frac{1}{2}}$ be positive square root of $\Sigma$ , i.e. $\Sigma = \Sigma^{\frac{1}{2}} \Sigma^{\frac{1}{2}}$ , and note that $\Sigma^{\frac{1}{2}}$ is also a positive definite matrix, moreover

$$
|\Sigma| = |\Sigma^{\frac{1}{2}} \Sigma^{\frac{1}{2}}| = |\Sigma^{\frac{1}{2}}|^2 \Longrightarrow |\Sigma^{\frac{1}{2}}| = |\Sigma|^{\frac{1}{2}}
$$

Let $z \in \mathbb{R}^k$ and

$$
z = \Sigma^{-\frac{1}{2}} (x - \mu) \Longleftrightarrow
x = \mu + \Sigma^{\frac{1}{2}} z
$$

Notice that positive definite matrix is ​​also a symmetric matrix, thus

$$
z^T = (x - \mu)^T (\Sigma^{-\frac{1}{2}})^T = (x - \mu)^T \Sigma^{-\frac{1}{2}}
$$

And the Jacobian determinant of this transformation is

$$
|\text{det}(\Sigma^{\frac{1}{2}})| = |\Sigma|^{\frac{1}{2}} , \text{ so } dx = |\Sigma|^{\frac{1}{2}} dz
$$

Then

$$
\begin{align*}
\int_{\mathbb{R}^k} f(x) dx
&= \int_{\mathbb{R}^k} \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \text{exp}\left(-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right) dx \\
&= \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \int_{\mathbb{R}^k} \text{exp}\left(-\frac{1}{2} z^T z \right) |\Sigma|^{\frac{1}{2}} dz \\
&= \frac{1}{(2\pi)^{\frac{k}{2}}} \int_{\mathbb{R}^k} \text{exp}\left(-\frac{1}{2} \sum_{i=1}^{k} z_i^2 \right) dz \\
&= \frac{1}{(2\pi)^{\frac{k}{2}}} \prod_{i=1}^{k} \int_{-\infty}^{\infty} e^{-\frac{z_i^2}{2}} dz
\end{align*}
$$

One-dimensional Gaussian integrals tell us

$$
\int_{-\infty}^{\infty} e^{-\frac{x^2}{2}} dx = \sqrt{2\pi}
$$

Thus

$$
\int_{\mathbb{R}^k} f(x) dx
= \frac{1}{(2\pi)^{\frac{k}{2}}} \left(\sqrt{2\pi} \right)^k
= 1
$$


## Problem 2

Let $A, B$ be $n$ -by- $n$ matrices and $x$ be a $n$ -by- $1$ vector.

### (a) Show that $\frac{\partial}{\partial A} \text{trace}(AB) = B^T$

Obviously,

$$
\text{tr}(AB) = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}b_{ji}
$$

And,

$$
\frac{\partial}{\partial a_{kl}} \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}b_{ji}
= \frac{\partial}{\partial a_{kl}} a_{kl}b_{lk} = b_{lk}
$$

So,

$$
\frac{\partial}{\partial A} \text{tr}(AB)
= \left[\frac{\partial}{\partial a_{ij}} \text{tr}(AB) \right]_{n \times n}
= \left[b_{ji} \right]_{n \times n}
= B^T
$$

### (b) Show that $x^TAx = \text{trace}(xx^TA)$

#### Lemma

Let $A$ be $m \times n$ matric, B be $n \times m$, then $\text{tr}(AB) = \text{tr}(BA)$

> Noticed that
>
> $$
> \text{tr}(AB)
> = \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}b_{ji}
> = \sum_{i=1}^{n} \sum_{j=1}^{m} b_{ji}a_{ij}
> = \text{tr}(BA)
> $$

by Lemma,

$$
\text{tr}(xx^TA) = \text{tr}(x^TAx)
$$

Here $x^T \in M_{1 \times n}, A \in M_{n \times n}, x \in M_{n \times 1}$, so $x^TAx \in \mathbb{R}$

So,

$$
\text{tr}(xx^TA) = \text{tr}(x^TAx) = x^TAx
$$

### (c) Derive the maximum likelihood estimators for a multivariate Gaussian

Let $x_1, ..., x_n \in \mathbb{R^k}$ are i.i.d. samples from

$$
X \sim N(\mu, \Sigma)
$$

where $\mu \in \mathbb{R}^k$ and $\Sigma$ is a $k \times k$ positive definite covariance matrix

The likelihood function is

$$
L(\mu, \Sigma) = \prod_{i=1}^{n} \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \text{exp}\left(-\frac{1}{2} (x_i-\mu)^T \Sigma^{-1} (x_i-\mu) \right)
$$

log-likelihood:

$$
\ell(\mu, \Sigma) = -\frac{nk}{2}\ln(2\pi) - \frac{n}{2}\ln|\Sigma| - \frac{1}{2} \sum_{i=1}^{n} (x_i-\mu)^T \Sigma^{-1} (x_i-\mu)
$$

#### 1. MLE for $\mu$

Differentiate $\ell$ with $\mu$ ,

$$
\begin{align*}
\frac{\partial \ell}{\partial \mu}
&= -\frac{1}{2} \sum_{i=1}^{n} \frac{\partial}{\partial \mu} (x_i-\mu)^T \Sigma^{-1} (x_i-\mu) \\
&= -\frac{1}{2} \sum_{i=1}^{n} \frac{\partial}{\partial \mu} \left(x_i^T \Sigma^{-1} x_i - 2 x_i^T \Sigma^{-1} \mu + \mu^T \Sigma^{-1} \mu \right) \\
&= -\frac{1}{2} \sum_{i=1}^{n} \frac{\partial}{\partial \mu} \left(-2 x_i^T \Sigma^{-1} \mu + \mu^T \Sigma^{-1} \mu \right)
\end{align*}
$$

Where,

$$
\frac{\partial}{\partial \mu} \left(-2 x_i^T \Sigma^{-1} \mu \right)
= -2 \Sigma^{-1} x_i
$$

$$
\frac{\partial}{\partial \mu} \left(\mu^T \Sigma^{-1} \mu \right)
= 2 \Sigma^{-1} \mu
$$

Hence,

$$
\frac{\partial \ell}{\partial \mu}
= \Sigma^{-1} \sum_{i=1}^{n} (x_i - \mu)
$$

Let $\frac{\partial \ell}{\partial \mu} = 0$ , have

$$
\sum_{i=1}^{n} (x_i - \mu) = 0 \\
\Longrightarrow \mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

Therefore,

$$
\mu_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

#### 2. MLE for $\Sigma$

By Result in (b),

$$
(x-\mu)^T \Sigma^{-1} (x-\mu) = \text{tr}\left((x-\mu) (x-\mu)^T \Sigma^{-1} \right)
$$

And by Result in (a),

$$
\frac{\partial}{\partial \Sigma^{-1}} \text{tr}\left((x-\mu) (x-\mu)^T \Sigma^{-1} \right)
= (x-\mu) (x-\mu)^T
$$

By some Compute, have

$$
\frac{\partial}{\partial A} f(A^{-1}) = -A^{-T} \left(\frac{\partial}{\partial A^{-1}} f(A^{-1}) \right) A^{-T}
$$

So, we have

$$
\frac{\partial}{\partial \Sigma} \text{tr}\left((x-\mu) (x-\mu)^T \Sigma^{-1} \right)
= -\Sigma^{-1} (x-\mu) (x-\mu)^T \Sigma^{-1}
$$

Back to the question, we differentiate $\ell$ with $\Sigma$ ,

$$
\frac{\partial \ell}{\partial \Sigma}
= -\frac{n}{2} \Sigma^{-1} + \frac{1}{2} \Sigma^{-1} \left(\sum_{i=1}^{n} (x_i-\mu) (x_i-\mu)^T \right) \Sigma^{-1}
$$

Let $\frac{\partial \ell}{\partial \Sigma} = 0$ , have

$$
\Sigma^{-1} \left(\sum_{i=1}^{n} (x_i-\mu) (x_i-\mu)^T \right) \Sigma^{-1} - n \Sigma^{-1} = 0 \\
\Longrightarrow
\sum_{i=1}^{n} (x_i-\mu) (x_i-\mu)^T = n \Sigma
$$

Therefore,

$$
\Sigma_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} (x_i-\mu) (x_i-\mu)^T
$$

#### note

the likelihood at these values is a maximum because $\ell$ is concave in $\mu$ and $\Sigma$ and the stationary conditions above yield the global maximum.

#### Final Answer

$$
\mu_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i , 
\Sigma_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} (x_i-\mu) (x_i-\mu)^T
$$


## Problem 3

Unanswered Questions

1. How does MLE change if the data are assumed independent but not identically distributed?
2. How do we compute MLE when some data points are missing?

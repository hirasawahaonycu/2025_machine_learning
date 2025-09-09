## Problem 1

Let $\alpha$ be learning rate.

By gradient descent algorithm,

$$\theta^1 = \theta^0 - \alpha \nabla_{\theta}\text{Loss}$$

Consider MSE loss of SGD:

$$\text{Loss} = \frac{1}{2}\| y - h(x_1, x_2)\|^2$$

Thus,

$$
\begin{align*}
\theta^1 &= \theta^0 - \frac{1}{2} \alpha \nabla_{\theta} (y - h(x_1, x_2))^2 \\
&= \theta^0 + \alpha (y - h(x_1, x_2)) \nabla_{\theta} h(x_1, x_2)
\end{align*}
$$

For the sigmoid function

$$ \sigma(x) = \frac{1}{1+e^{-x}}, $$

we have

$$ \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) $$

By $ h(x_1, x_2) = \sigma(b + w_1x_1 + w_2x_2) $, we have

$$
\begin{align*}
\nabla_{\theta} h(x_1, x_2) &= \sigma'(b + w_1x_1 + w_2x_2) \cdot \nabla_{\theta}(b + w_1x_1 + w_2x_2) \\
&= \sigma(b + w_1x_1 + w_2x_2) (1 - \sigma(b + w_1x_1 + w_2x_2)) \cdot (1, x_1, x_2) \\
&= h(x_1, x_2) (1 - h(x_1, x_2)) \cdot (1, x_1, x_2)
\end{align*}
$$

So,

$$ \theta^1 = \theta^0 + \alpha \, h(x_1, x_2) \, (y - h(x_1, x_2)) \, (1 - h(x_1, x_2)) \cdot (1, x_1, x_2) $$

Given $(x_1, x_2, y) = (1, 2, 3)$ and $\theta^0 = (4, 5, 6)$,

$$ h(1, 2) = \sigma(21) $$

Thus,

$$ \theta^1 = (4, 5, 6) + \alpha \, \sigma(21) \, (3 - \sigma(21)) \, (1 - \sigma(21)) \cdot (1, 1, 2) $$


## Problem 2
### (a)

Sigmoid function:

$$ \sigma(x) = \frac{1}{1+e^{-x}} $$

For $k = 1$,

$$ \sigma(x) \cdot (1 + e^{-x}) = 1 $$
$$ \Rightarrow \quad \sigma'(x) (1 + e^{-x}) - \sigma(x)e^{-x} = 0 $$
$$ \Rightarrow \quad \sigma'(x) = \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}} $$
$$ \Rightarrow \quad \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) $$

For $k = 2$,

$$ \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) $$
$$ \Rightarrow \quad \sigma''(x) = \sigma'(x) \cdot (1 - 2 \sigma(x)) $$
$$ \Rightarrow \quad \sigma''(x) = \sigma(x) \cdot (1 - \sigma(x)) \cdot (1 - 2 \sigma(x)) $$

For $k = 3$,

$$
\begin{align*}
\sigma''(x) &= \sigma(x) \cdot (1 - \sigma(x)) \cdot (1 - 2 \sigma(x)) \\
&= \sigma(x) - 3 (\sigma(x))^2 + 2 (\sigma(x))^3
\end{align*}
$$
$$ \Rightarrow \quad \sigma'''(x) = \sigma'(x) - 6 \sigma(x) \sigma'(x) + 6 (\sigma(x))^2 \sigma'(x) $$
$$ \Rightarrow \quad \sigma'''(x) = \sigma(x) \cdot (1 - \sigma(x)) \cdot (1 - 6 \sigma(x) + 6 (\sigma(x))^2) $$

So,

$$
\begin{align*}
\frac{d}{dx}\sigma(x) &= \sigma(x) \, (1 - \sigma(x)) \\
\frac{d^2}{dx^2}\sigma(x) &= \sigma(x) \, (1 - \sigma(x)) \, (1 - 2 \sigma(x)) \\
\frac{d^3}{dx^3}\sigma(x) &= \sigma(x) \, (1 - \sigma(x)) \, (1 - 6 \sigma(x) + 6 \sigma^2(x)) 
\end{align*}
$$


### (b)

Sigmoid function:

$$ \sigma(x) = \frac{1}{1+e^{-x}} $$

Hyperbolic function:

$$ \text{tanh}(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} $$

Noticed that

$$
\begin{align*}
\text{tanh}(x) + 1 &= \frac{2e^{x}}{e^{x} + e^{-x}} \\
&= \frac{2}{1 + e^{-2x}} \\
&= 2 \sigma(2x)
\end{align*}
$$

So,

$$ \text{tanh}(x) = 2 \sigma(2x) - 1 $$


## Problem 3

1. Why does adding a nonlinear layer after each linear layer make the model better?

2. How can we determine the number of layers and the number of neurons in a neural network?

3. Why do we train the data in batches, how does this affect the training results?

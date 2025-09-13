## Problem 1


### Define the network:

Input layer:

$$ a^{[1]} = x \in \mathbb{R^{n_1}} $$

Hidden layer:

$$
\begin{align*}
    z^{[l]} &= W^{[l]}a^{[l-1]} + b^{[l]} \in \mathbb{R^{n_l}}, \\
    a^{[l]} &= \sigma(z^{[l]}) \in \mathbb{R^{n_l}}
\end{align*}
$$

Output layer:

$$
\begin{align*}
    z^{[L]} &= W^{[L]}a^{[L-1]} + b^{[L]} \in \mathbb{R}, \\
    a^{[L]} &= \sigma(z^{[L]}) \in \mathbb{R}
\end{align*}
$$


### Algorithm:

#### 1. Forward propogation

Compute all $z^{[l]}$ and $a^{[l]}$ for $l = 2, ..., L$ .

#### 2. Backward propogrtion

Define

$$ \delta^{[l]} = \frac{\partial a^{[L]}}{\partial z^{[l]}}, \text{ for } l = 2, ..., L. $$

For $l = L$,

$$
\delta^{[L]}
= \frac{\partial a^{[L]}}{\partial z^{[L]}}
= \sigma'(z^{[L]})
$$

For $l = L-1, L-2, ..., 2$,

$$
\delta^{[l]}
= \frac{\partial a^{[L]}}{\partial z^{[l]}} 
= \sigma'(z^{[l]}) \circ \left[(W^{[l+1]})^T \delta^{[l+1]} \right]
$$

#### 3. Compute $\nabla a^{[L]}(x)$

Note that

$$
z^{[2]} = W^{[2]}x + b^{[2]}
$$

Thus,

$$
\nabla_x a^{[L]}(x)
= \frac{\partial a^{[L]}}{\partial x}
= (W^{[2]})^{T} \delta^{[2]}
$$

<!-- 
Note that

$$
\nabla a^{[L]}(x) = (
    \frac{\partial a^{[L]}}{\partial W^{[1]}},
    \frac{\partial a^{[L]}}{\partial b^{[1]}},
    ...,
    \frac{\partial a^{[L]}}{\partial W^{[l]}},
    \frac{\partial a^{[L]}}{\partial b^{[l]}},
    ...,
    \frac{\partial a^{[L]}}{\partial W^{[L]}},
    \frac{\partial a^{[L]}}{\partial b^{[L]}}
),\
l = 2, ..., L.
$$

And for all $l = 2, ..., L$,

$$
\frac{\partial a^{[L]}}{\partial b^{[l]}}
= \frac{\partial a^{[L]}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial b^{[l]}}
= \delta^{[l]}
$$

$$
\frac{\partial a^{[L]}}{\partial W^{[l]}}
= \frac{\partial a^{[L]}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial W^{[l]}}
= \delta^{[l]} (a^{[l-1]})^{T}
$$
-->


## Problem 2

In the backpropagation, each $\delta^{[l]}$ is multiplied by the previous $\delta^{[l+1]}$. However, if during a single epoch, x becomes either extremely small or extremely large, the subsequent values of x will likely remain very small or very large. This can cause the model to stop learning entirely or lead to excessively large parameter updates. How to solve this problem?

Answer of ChatGPT:
1. Proper weight initialization: Xavier or He initialization keeps signals stable during forward and backward propagation.
2. Use suitable activation functions: ReLU or Leaky ReLU are less prone to vanishing gradients compared to sigmoid or tanh.
3. Gradient clipping: Limit the maximum norm of gradients to prevent exploding gradients.
4. Normalization techniques: Batch Normalization helps stabilize the distribution of activations across layers.
5. Adjust the learning rate: A smaller learning rate helps avoid excessively large updates.
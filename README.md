# Neural Networks And Deep Learning
Programming assignments from Coursera class https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome

***
### Logistic Regression
- $z = w^{T}*x + b \rightarrow a = \sigma(z)$
- X is $[n^{x}, m]$
- $\hat{y}^{(i)} = a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 - e^{-z^{(i)}}}$

***
### Shallow Neural Network
- $(i)$ - training sample
- $[i]$ - layer
- a - activations $a^{[0]}, a^{[1]}, a^{[2]}$ - 2-LAYER NETWORK
    - $a^{[0]}$ is $X$ and has dimension $[n^{x}, m]$ - INPUT
    - $a^{[1]}$ has dimension $[number_{units}, 1]$ - HIDDEN
    - $a^{[2]}$ is $\hat{y}$ - OUTPUT
- 1st [layer], 1st node - $z^{[1]}_{1} = w^{[1]T}_{1}*x + b^{[1]}_{1} \rightarrow a^{[1]}_{1} = \sigma(z^{[1]}_1)$
- Sigmoid activation
- ReLU - Rectified Linear Unit - learning faster in most of the cases
- Leaky ReLU - slope smaller for z < 0 -> max(0.01 * z, z)
- sigmoid activation function used for binary classification






***
[Math in Markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

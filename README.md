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
### Deep L-Layer Neural Network
- Forward
    - $Z^{[l]} = W^{[l]T} * A^{[l-1]} + b^{[l]}$
    - $A^{[l]} = g^{[l]}(Z^{[l]})$ [note: $A^{[L]} = g^{[L]}(Z^{[L]})$]
- Backward
    - $\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = \frac{1}{m} (a^{[2](i)} - y^{(i)})$
    - $\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T} $
    - $\frac{\partial \mathcal{J} }{ \partial b_2 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}$
    - $\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2}) $
    - $\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T $
    - $\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}$
- Backward
    - $dZ^{[L]} = A^{[L]} - Y$
    - $dW^{[L]} = \frac{1}{m} dZ^{[L]} A^{[L-1]T}$
    - $db^{[L]} = \frac{1}{m}np.sum(dZ^{[L]}, axis=1, keepdims=True)$
    - $dA^{[L-1]} = W^{[L]T}dZ^{[L]}*g\rq^{[L-1]}(Z^{[L-1]})$

- Note that $*$ denotes elementwise multiplication.
- The notation you will use is common in deep learning coding:
    - dW1 = $\frac{\partial \mathcal{J} }{ \partial W_1 }$
    - db1 = $\frac{\partial \mathcal{J} }{ \partial b_1 }$
    - dW2 = $\frac{\partial \mathcal{J} }{ \partial W_2 }$
    - db2 = $\frac{\partial \mathcal{J} }{ \partial b_2 }$





***
[Math in Markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

# Neural Networks And Deep Learning
Programming assignments from Coursera class https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome

- Great notes taken [here](https://aman.ai/coursera-dl/)

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
### Regularization
Over-fitting/high-variance problems.
- Logistic Regression
    - $J(w, b) = \frac{1}{m}\displaystyle\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}||w||_2^2$
    - $\lambda$ - regularization parameter
    - $L_2$ regularization $||w||_2^2 = \displaystyle\sum_{j=1}^{n_x}w_j^2 = w^Tw$ (Euclid form)
    - $L_1$ regularization $||w||_1 = \displaystyle\sum_{j=1}^{n_x}|w_j|$
- Neural Network
    - $J(w^{[i]}, b^{[i]}, ..., w^{[L]}, b^{[L]}) = \frac{1}{m}\displaystyle\sum_{i=1}^{m}L(\hat{y^{(i)}}, y^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L}||w^{[l]}||_F^2$
    - "Frobenius norm" $||w^{[l]}||_F^2 = \displaystyle\sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{[l-1]}}||w_{ij}^{[l]}||^2$
    - $dw^{[l]} = w^{[l]} + \frac{\lambda}{m}w^{[l]}$
    - $w^{[l]} = w^{[l]} - \alpha*dw^{[l]}$
    - L2 norm regularization is often called "weight decay", decreasing $W^{l}$ by $\frac{\alpha\lambda}{m}$
- Dropout Regularization
    - (most common) Iverted Dropout
        - keep probability - $keepprob$
        - zero out different nurons in different layers
        - (for example, in layer 3)
            - $d3 = np.random.randn(a3.space[0], a3.space[1]) < keepprob$
            - $a3 */ d3$
            - $a3 /= keepprob$
- Early Stopping
    - Stop at the point where where is a potential dev error increase all the while cost function is trending down
    - Combines optimizing cost function with over-fitting
        - This is ont good, trying to do 2 tasks at once
        - Regularization is better choice but it likely takes more time and resources

***
### Optimization problem (speeding up the training)
- Normalizing training sets
    - whenranges of features are in different scale, optimizing time to train
    - Two step process
        - Subtract/zero-out mean: $\mu = \frac{1}{m}\displaystyle\sum_{i=1}^{m}x^{(i)}$ and $x = x - \mu$
        - Normalize variances: $\sigma^{2} = \frac{1}{m}\displaystyle\sum_{i=1}^{m}{X^{(i)}}^{2}$ and $x = x / \sigma$
    - if normalizing, do it both train and test data set
    - faster progresion of gradient

- Vanishing/exploding gradients
    - weight initialization
        - $W^{[l]} = np.random.randn*(A.shape[0]. A.shape[1]) * np.sqrt(\frac{1}{n^{[l-1]}})$ - where $n$ is number of neurons
        - Variation $frac{1}{n}$ for ReLU is better as $\sqrt\frac{2}{n}$
        - Variance for $tanh()$ is $\sqrt{\frac{1}{n^{[l-1]}}}$
        - Xavier variance $\sqrt\frac{1}{n^{[l-1]}+n^{[l]}}$

- Gradient Checking
    - use in dev only, not in training
    - turn $(W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]})$ into $\theta$
    - and cost function $J(W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}) = J(\theta)$
    - expect $\epsilon = 10^{-7}$
    - for each $i$:
        - $d\theta_{approx} = \frac{J(\theta{1}, ..., \theta_{1+\epsilon}, ...) - J(theta_{1}, ..., \theta_{1-\epsilon}, ...)}{2\epsilon}$
        - $d\theta[i] = \frac{\partial J}{\partial \theta[i]}$
        - $\frac{||d\theta_{approx} - d\theta||_2}{||d\theta_{approx}||_2 + ||d\theta||_2}$
            - $\approx \epsilon$ : GREAT!
            - otherwise investigate
            - $10^{-3}$ : WORRY$

***
### Optimization Algorithms
- Batch vs miniu-batch gradient decent
    - $X^{\{i\}}$ and $Y^{\{i\}}$ - $i^{th}$ mini-batch input and output
    - one cycle of processing (forward, backward propagation) of all mini-batches (say 1,000 records out of 5M) is called "1 epoch"
    - stochastic gradient decent: size of mini-batch is 1, $(X^{\{1\}}, Y^{\{1\}}) = (x^{(1)}, y^{(1)}), ...$
        - does not copnverge well
    - minhi-batch should be in between 1 and 'm'
    - mini-batch size
        - smal training set, < 1000 training set : go batch gradient descent
        - otherwise go in mini-batch increment in exponent of 2: 64, 128, ..., 512; for very large go even 1024
            - mini-batch size has to fit into CPU/GPU and memory
- Expotentially weighted (moving) averages
    - $V_t = \beta*V_{t-1} + (1-\beta)*\theta_t$
    - $V_{\theta} = \beta*V_{\theta} + (1-\beta)*\theta_t$ (keep updating $V_{\theta}$ latest value)
        - just keep updating, does not require large amount of memory to store all the data points amnd compute straight average)
    - most common value is $\beta = 0.9$ whichis average over 10 time periods (like temperature over days)
    - Bias Corection $V_t = \frac{V_t}{1-\beta^{t}}$
    - It is usually not in used in practice due to the fact that over number of points (like 10 points in $\beta = 0.9$, it stabilizes)
- Gradient descent with momentum
    - $v_{dW} = \beta*V_{dW} + (1 - \beta)*dW$
    - $v_{db} = \beta*V_{db} + (1 - \beta)*db$
    - and use it in 
        - $W = W - \alpha * v_{dW}$
        - $b = b - \alpha * v_{db}$
    - (think of it as $v_{dW}, v_{db}$ are velocity and $\beta$ as friction)
- RMSprop
    - RMS - root means square
    - $S_{dW} = \beta S_{dW} + (1 - \beta) {dW}^2$ - element-wise square
    - $S_{db} = \beta S_{db} + (1 - \beta) {db}^2$ - element-wise square
    - and use it in 
        - $W = W - \alpha \frac{dW}{\sqrt{S_{dW}}}$
        - $b = b - \alpha \frac{db}{\sqrt{S_{db}}}$
- Adam optimization
    - "adaptive moment estimation" - combines Momentum and RSMprop optimization
    - commonly used gradient optimization algorithm for neural networks and different architectures
    - initialize $V_{dW} = 0, S_{dW} = 0, V_{db} = 09, S_{db} = 0$
    - for iteration $t$:
        - momentum: $V_{dW} = \beta_{1} V_{dW} + (1 - \beta_{1}) dW, V_{db} = \beta_{1} V_{db} + (1 - \beta_{1}) db$
        - RMSprop: $S_{dW} = \beta_{2} S_{dW} + (1 - \beta_{2}) dW, S_{db} = \beta_{2} S_{db} + (1 - \beta_{2}) db$
        - bias correction:
        - $V_{dW}^{corrected} = \frac{V_{dW}}{1-\beta_{1}^{t}}, V_{db}^{corrected} = \frac{V_{db}}{1-\beta_{1}^{t}}$
        - $S_{dW}^{corrected} = \frac{S_{dW}}{1-\beta_{2}^{t}}, S_{db}^{corrected} = \frac{S_{db}}{1-\beta_{2}^{t}}$
        - calculate:
        - $W = W - \alpha \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}} + \epsilon}$
        - $b = b - \alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}} + \epsilon}$
    - Hyperparameters choice
        - $\alpha$ - needs to be tuned
        - $\beta_{1} = 0.9$ - beta moment, moving average for momentum is commonly set to 0.9
        - $\beta_{2} = 0.999$ - second moment, moving acverage squared, set by authors of the algorithm
        - $\epsilon = 10^{-8}$ - does not impct performance much at all, often not used
    - Learning rate decay
        - slowly reduce leraning rate over time
        - $\alpha = \frac{1}{1 - decay\_rate * epoch\_num} * \alpha_{0}$
        - another methods:
            - exponential decay: $\alpha = 0.95^{epoch\_num} \alpha_{0}$ - 0.95 chosen initially
            - $\alpha = \frac{k}{\sqrt{epoch\_num}} \alpha_{0}$
            - $\alpha = \frac{k}{\sqrt{t}} \alpha_{0}$ : t=mini batch number
        - manual decay can also be used based on observation of slow gradient
        - not oftenly used, there are better ways

***
[Math in Markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

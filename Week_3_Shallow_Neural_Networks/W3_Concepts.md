# Week 3: Shallow Neural Networks

## Objective

This week, we move from a single neuron (Logistic Regression) to a complete neural network with one hidden layer. The focus is on understanding the architecture, the forward and backward propagation steps, the importance of activation functions, and the critical concept of random initialization.

---

## 1. Architecture of a 2-Layer Neural Network

A shallow neural network typically refers to a network with a single hidden layer. It's the simplest form of a "deep" network and consists of three parts:

* **Input Layer (`a[0]`):** This is not counted as a layer. It's where the feature vector `X` enters the network.
* **Hidden Layer (`a[1]`):** A layer of neurons that is not directly connected to the input or output. Its job is to learn intermediate representations of the data. The parameters for this layer are `W[1]` and `b[1]`.
* **Output Layer (`a[2]`):** The final layer that produces the prediction `a[2]`. Its parameters are `W[2]` and `b[2]`.

The flow of information goes from the input layer, through the hidden layer, to the output layer to make a final prediction.

---

## 2. Forward Propagation

Forward propagation is the process of calculating the network's output, `A[2]`, starting from the input `X`. For a vectorized implementation across `m` examples, the steps are:

1.  **Calculate the hidden layer's linear combination:**
    * `Z[1] = W[1]A[0] + b[1]` (Note: `A[0]` is the input `X`).
    * The shape of `W[1]` is `(n[1], n[0])`, where `n[1]` is the number of neurons in the hidden layer and `n[0]` is the number of input features.
    * The shape of `Z[1]` will be `(n[1], m)`.

2.  **Apply the hidden layer's activation function:**
    * `A[1] = g[1](Z[1])`, where `g[1]` is the chosen activation function (e.g., ReLU or tanh).
    * The shape of `A[1]` is also `(n[1], m)`.

3.  **Calculate the output layer's linear combination:**
    * `Z[2] = W[2]A[1] + b[2]`.
    * The shape of `W[2]` is `(n[2], n[1])`, where `n[2]` is the number of output neurons (which is 1 for binary classification).
    * The shape of `Z[2]` will be `(n[2], m)`, or `(1, m)`.

4.  **Apply the output layer's activation function:**
    * `A[2] = g[2](Z[2])`, where `g[2]` is typically the sigmoid function for binary classification.
    * The shape of `A[2]` is `(1, m)`. This is our final vector of predictions.

---

## 3. Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.

### Why Non-Linearity is Necessary
If we only used linear activation functions (or no activation functions), our neural network, no matter how many layers it had, would just be equivalent to a single linear model. The non-linearity is what gives deep networks their power.

### Common Activation Functions

* **Sigmoid:** `a = 1 / (1 + exp(-z))`
    * **Pros:** Squashes output to a range of [0, 1], which is useful for the output layer in binary classification.
    * **Cons:** Can lead to the "vanishing gradient" problem, which slows down training. Not recommended for hidden layers.
* **Tanh (Hyperbolic Tangent):** `a = tanh(z)`
    * **Pros:** Squashes output to a range of [-1, 1]. Its output is zero-centered, which can help center the data for the next layer and often makes learning faster than sigmoid.
    * **Cons:** Also suffers from the vanishing gradient problem, although it is generally preferred over sigmoid for hidden layers.
* **ReLU (Rectified Linear Unit):** `a = max(0, z)`
    * **Pros:** Currently the most popular activation function for hidden layers. It's computationally simple and helps significantly with the vanishing gradient problem, allowing for faster training.
    * **Cons:** The derivative is zero for negative inputs, which can cause some neurons to "die" and stop updating.
* **Leaky ReLU:** `a = max(0.01*z, z)`
    * A variation of ReLU that attempts to solve the "dying neuron" problem by allowing a small, non-zero gradient for negative inputs.

---

## 4. Backward Propagation

Backward propagation calculates the gradient of the cost function with respect to each parameter (`W[1]`, `b[1]`, `W[2]`, `b[2]`). These gradients are then used by the gradient descent algorithm to update the parameters.

For a 2-layer network, the vectorized equations are:

$$ dZ^{[2]} = A^{[2]} - Y $$
$$ dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T} $$
$$ db^{[2]} = \frac{1}{m} \text{np.sum}(dZ^{[2]}, \text{axis=1, keepdims=True}) $$
$$ dZ^{[1]} = W^{[2]T} dZ^{[2]} * g'^{[1]}(Z^{[1]}) $$
$$ dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T} $$
$$ db^{[1]} = \frac{1}{m} \text{np.sum}(dZ^{[1]}, \text{axis=1, keepdims=True}) $$

*The `*` in the `dZ[1]` calculation represents element-wise multiplication. `g'[1]` is the derivative of the activation function used in the hidden layer.*

---

## 5. The Importance of Random Initialization

How we initialize our parameters is critical.

* **The Problem:** If we initialize all weights `W` to zero, all neurons in the hidden layer are identical. During backpropagation, they will all receive the same update and will continue to be identical in subsequent iterations. This "symmetry" prevents the neurons from learning different features, making the hidden layer useless.
* **The Solution:** We must **break the symmetry** by initializing the weights to small random numbers. Biases, however, can be safely initialized to zero as the random weights are sufficient to break the symmetry.

A common practice is:
```python
# 0.01 is used to keep the initial weights small
W1 = np.random.randn((n_h, n_x)) * 0.01
b1 = np.zeros((n

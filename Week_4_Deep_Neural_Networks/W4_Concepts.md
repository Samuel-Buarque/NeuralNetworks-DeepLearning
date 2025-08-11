# Week 4: Deep Neural Networks

## Objective

This week, we generalize the concepts from a shallow (2-layer) network to a deep L-layer neural network. The focus is on establishing a consistent notation, understanding how forward and backward propagation scale to multiple layers, and building an intuition for why deep representations are so effective.

---

## 1. Notation for an L-Layer Network

When moving to deep networks, having a consistent notation is crucial.

* `L`: The total number of layers in the network (not including the input layer).
* `n[l]`: The number of neurons (or units) in layer `l`.
    * `n[0]` is the number of features in the input layer.
    * `n[L]` is the number of units in the output layer.
* `g[l]`: The activation function used in layer `l`.
* `W[l]`: The weights matrix for layer `l`. It has a shape of `(n[l], n[l-1])`.
* `b[l]`: The bias vector for layer `l`. It has a shape of `(n[l], 1)`.
* `Z[l]`: The linear output of layer `l`.
* `A[l]`: The activation output of layer `l`, where `A[l] = g[l](Z[l])`.
    * The input `X` is denoted as `A[0]`.
    * The final prediction `ŷ` is the activation of the final layer, `A[L]`.

---

## 2. Forward Propagation in a Deep Network

Forward propagation in a deep network is a chain of computations repeated for each layer, from `l=1` to `L`. A `for` loop is generally required to implement this.

The general rule for any layer `l` is:
1.  **Calculate the linear combination:**
    $$ Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} $$
2.  **Calculate the activation:**
    $$ A^{[l]} = g^{[l]}(Z^{[l]}) $$

The output of one layer, `A[l-1]`, becomes the input for the next layer, `l`. During this process, it's essential to cache the `Z[l]` values for each layer, as they are needed for backpropagation.

### Getting Matrix Dimensions Right

Debugging neural networks often comes down to fixing matrix dimension mismatches. A good rule of thumb for any layer `l`:

* `W[l]`: `(n[l], n[l-1])`
* `b[l]`: `(n[l], 1)`
* `Z[l]`, `A[l]`: `(n[l], m)` where `m` is the number of examples.
* `dW[l]` must have the same dimension as `W[l]`.
* `db[l]` must have the same dimension as `b[l]`.

---

## 3. Backward Propagation in a Deep Network

Similarly, backpropagation involves chaining derivative calculations from the last layer (`L`) back to the first (`l=1`).

The general rule for any layer `l` is:

1.  **Calculate the derivative of the linear output:**
    $$ dZ^{[l]} = dA^{[l]} * g'^{[l]}(Z^{[l]}) $$
    * The `*` denotes element-wise multiplication. `g'[l]` is the derivative of the activation function of layer `l`.

2.  **Calculate the gradients for weights and biases:**
    $$ dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} $$
    $$ db^{[l]} = \frac{1}{m} \text{np.sum}(dZ^{[l]}, \text{axis=1, keepdims=True}) $$

3.  **Propagate the error to the previous layer:**
    $$ dA^{[l-1]} = W^{[l]T} dZ^{[l]} $$

The process starts by calculating `dA[L]`, the derivative of the cost function with respect to the final activation `A[L]`. Then, these three steps are repeated in a loop from `l=L` down to `1`.

---

## 4. Why Deep Representations?

Why do deep networks perform so well? The intuition is that they build a hierarchy of features, learning progressively more complex concepts layer by layer.

* **Simple to Complex:** The first layers might learn to detect simple features like edges or color gradients in an image. Subsequent layers combine these simple features to learn more complex concepts like textures, patterns, or parts of an object (e.g., an eye, a nose). The final layers can then combine these parts to recognize a complete object (e.g., a face).
* **Circuit Theory Analogy:** From a circuit theory perspective, a deep network can compute certain functions with an exponentially smaller number of neurons compared to a shallow network. This suggests that deep networks have a more efficient architecture for representing complex functions.

When building a solution, it's often best practice to start simple (e.g., with Logistic Regression or a shallow network) and only increase the number of layers (`L`) as needed, treating it as a hyperparameter to be tuned.

---

## 5. Parameters vs. Hyperparameters

It's crucial to distinguish between the two:

* **Parameters:** These are the values that the model learns on its own during training. In a neural network, the parameters are the weights `W` and biases `b` for every layer.
* **Hyperparameters:** These are the configuration settings that we, the engineers, must choose to guide the learning process. They are not learned from the data. Key examples include:
    * **Learning Rate (`alpha`):** The most important hyperparameter.
    * **Number of Iterations:** How many times to run gradient descent.
    * **Number of Hidden Layers (`L`):** The depth of the network.
    * **Number of Hidden Units (`n[l]`):** The width of each layer.
    * **Choice of Activation Function (`g[l]`):** (ReLU, Tanh, etc.)

The process of finding the best hyperparameters is a core part of applied machine learning, and we will explore techniques for this in later stages.

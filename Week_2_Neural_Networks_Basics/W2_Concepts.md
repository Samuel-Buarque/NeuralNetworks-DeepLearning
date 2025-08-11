# Week 2: Neural Network Basics - The Mechanics

## Objective

This document covers the core mechanics of a basic neural network, focusing on **Logistic Regression** as the foundational algorithm for binary classification. We will explore how a model learns from data using a cost function and gradient descent, and how to implement these operations efficiently using **vectorization**.

---

## 1. The Problem: Binary Classification

Binary classification is the task of categorizing an input into one of two possible classes (e.g., 0 or 1, True or False). For this week, we'll use the common example of "cat vs. no-cat" image classification.

### Key Notations

* `m`: The number of training examples in the dataset.
* `Nx`: The number of features in the input vector (e.g., total pixels in a flattened image).
* `X`: The input matrix of shape `(Nx, m)`, where each column is a single training example.
* `Y`: The output vector of shape `(1, m)`, containing the correct labels (0 or 1) for each example.

---

## 2. The Algorithm: Logistic Regression

For binary classification, we use Logistic Regression. It serves as a single neuron and is the fundamental building block for more complex networks.

The process for a single training example `x` involves two steps:

1.  **Linear Combination:** Calculate a linear combination of the inputs, weights (`w`), and a bias (`b`).
    $$ z = w^T x + b $$
2.  **Activation:** Pass the result `z` through a **sigmoid activation function** to squash the output into a probability between 0 and 1.
    $$ a = \sigma(z) = \frac{1}{1 + e^{-z}} $$
    * The output `a` represents the predicted probability that the label `y` is 1.

---

## 3. Measuring Performance: The Cost Function

To train our model, we first need to quantify how well it's performing.

* **Loss Function `L(a, y)`:** Measures the error for a **single training example**. While we could use mean squared error, it results in a non-convex optimization problem for logistic regression, meaning our optimization algorithm could get stuck in local minima. Instead, we use the **log loss** (or binary cross-entropy) function:
    $$ L(a, y) = - (y \log(a) + (1 - y) \log(1 - a)) $$
    This function heavily penalizes predictions that are both confident and wrong.

* **Cost Function `J(w, b)`:** Measures the average error across the **entire training set** of `m` examples. It is simply the average of the loss function over all examples:
    $$ J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(a^{(i)}, y^{(i)}) $$

Our goal is to find the parameters `w` and `b` that minimize this cost function `J(w, b)`.

---

## 4. The Learning Process: Gradient Descent

Gradient Descent is the optimization algorithm used to find the optimal `w` and `b`. It works by iteratively updating the parameters in the direction that most steeply decreases the cost function.

The process is as follows:
1.  Initialize parameters `w` and `b` (typically to zeros for logistic regression).
2.  Repeatedly execute the following update steps for a set number of iterations:
    $$ w := w - \alpha \cdot dw $$
    $$ b := b - \alpha \cdot db $$
    * `alpha` (`\alpha`) is the **learning rate**, a hyperparameter that controls the size of each step.
    * `dw` and `db` are the derivatives (or gradients) of the cost function with respect to `w` and `b`. They tell us the direction of the steepest ascent of the cost function, so we subtract them to move "downhill".

---

## 5. Calculating Updates: Forward & Backward Propagation

A single iteration of training involves two main phases:

1.  **Forward Propagation:**
    * This is the process of making predictions.
    * For the entire training set `X`, we compute `Z`, then `A`, and finally use `A` to calculate the overall cost `J`.
    * This flows from left-to-right through the computation graph.

2.  **Backward Propagation:**
    * This is the process of calculating the gradients (`dw`, `db`).
    * It starts with the final error (from the cost `J`) and works backward through the network, applying the chain rule of calculus to determine how much each parameter contributed to the error.
    * The key derivatives calculated are: `dZ`, `dW`, and `db`.

---

## 6. The Key to Efficiency: Vectorization

In deep learning, processing large datasets is standard. Using `for` loops to iterate over `m` training examples is computationally inefficient and will not scale.

**Vectorization** is the technique of performing calculations on entire matrices and vectors at once, rather than element-by-element. This leverages low-level hardware optimizations (SIMD) in CPUs and GPUs for massive performance gains.

* **Non-Vectorized (Slow):**
    ```python
    # (Conceptual pseudo-code)
    for i in range(m):
        z[i] = ...
    ```

* **Vectorized (Fast):**
    Using NumPy, we can compute `Z` for all `m` examples in a single operation.
    ```python
    # Z shape will be (1, m)
    Z = np.dot(w.T, X) + b
    ```

**Rule of thumb: Whenever possible, avoid explicit `for` loops and use NumPy's built-in functions.**

### Practical Notes on Python/NumPy

* **Broadcasting:** NumPy automatically expands smaller arrays to match the shape of larger arrays in certain operations. For example, when adding a `(1, 1)` bias `b` to a `(1, m)` matrix `Z`, NumPy effectively copies `b` `m` times to perform the element-wise addition.
* **Avoid Rank-1 Arrays:** Vectors created with a shape like `(m,)` can behave unpredictably (e.g., transpose operations may not work as expected). Always explicitly shape your vectors as `(m, 1)` or `(1, m)` using `reshape()` to prevent bugs.
* **Normalization:** Gradient descent often converges faster if input features are normalized (e.g., scaled to have a mean of 0 and a variance of 1).

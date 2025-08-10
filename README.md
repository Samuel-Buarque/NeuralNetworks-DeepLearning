# A Practical Guide to Neural Networks and Deep Learning

## Objective

This repository serves as a practical guide and knowledge base for the first course of the DeepLearning.AI Specialization, ["Neural Networks and Deep Learning"](https://www.coursera.org/learn/neural-networks-deep-learning) by Andrew Ng.

The goal is to distill the core concepts and provide working code implementations for each major topic. It is intended as an internal learning and onboarding resource for team members who are new to Deep Learning but have an intermediate programming background.

## How to Use This Repository

The content is organized into folders, with each folder corresponding to a week in the course. For each week, you will find:

1.  **A `Concepts.md` file:** This file contains a theoretical summary of the key concepts, intuitions, and mathematical foundations for that week's topics. It's recommended to start here.
2.  **A `.ipynb` Jupyter Notebook:** This notebook contains the practical Python code that implements the concepts discussed in the `Concepts.md` file. It includes code for building models, training them, and visualizing the results.

## Course Structure & Content

### [Week 1: Introduction to Deep Learning](./Week_1_Introduction_to_DL/)

* **Topics:** What a Neural Network is, understanding the drivers behind the current Deep Learning boom (Data, Computation, Algorithms), and the fundamentals of Supervised Learning.
* **Goal:** Build the high-level intuition for the "what" and "why" of Deep Learning.

### [Week 2: Neural Network Basics](./Week_2_Neural_Networks_Basics/)

* **Topics:** Binary Classification, Logistic Regression as a single-neuron network, cost and loss functions, the Gradient Descent algorithm, and an introduction to computation graphs and backpropagation.
* **Goal:** Implement and train your first "neuron" using Logistic Regression and understand the fundamental mechanics of how a model learns.

### [Week 3: Shallow Neural Networks](./Week_3_Shallow_Neural_Networks/)

* **Topics:** Building a neural network with a single hidden layer, non-linear activation functions (Sigmoid, Tanh, and ReLU), and the mechanics of forward and backward propagation in a 2-layer network. The importance of random weight initialization.
* **Goal:** Build a model capable of learning non-linear decision boundaries.

### [Week 4: Deep Neural Networks](./Week_4_Deep_Neural_Networks/)

* **Topics:** Scaling up to an L-layer ("deep") neural network, understanding the power of depth, generalizing forward and backward propagation for any architecture, and managing hyperparameters vs. parameters.
* **Goal:** Build a general set of functions to create and train deep neural networks of arbitrary depth.

# Week 1 – Introduction to Neural Networks and Deep Learning

This week introduces the fundamental concepts behind deep learning and neural networks, following the structure of Andrew Ng's course.  
The focus is on understanding what neural networks are, how supervised learning fits into them, and why deep learning has become so prominent.

---

## 1. What is a Neural Network?

A neural network is inspired by the way the brain processes information.  
At its simplest, a **single neuron** performs a weighted sum of the inputs and applies an activation function to produce an output.

- **Perceptron model**:  
  - Calculates `w ⋅ x + b` and outputs a binary result depending on a threshold.
  - Works for both real-valued and boolean inputs.
  - Limitation: output changes abruptly — small weight/bias adjustments can completely flip the result.

- **Logistic (Sigmoid) neuron**:  
  - Replaces the hard threshold with a smooth sigmoid curve, producing values between 0 and 1.
  - More robust to small changes in weights or biases.

- **Modern activation functions**:  
  - **ReLU** (Rectified Linear Unit) is now widely used, improving training speed and helping avoid vanishing gradients.
  - Hidden layers automatically learn internal representations from the input features.

---

## 2. Supervised Learning with Neural Networks

Supervised learning means we have **input–output pairs** `(X, Y)` and want to learn a mapping from X to Y.  
Neural networks can take many forms depending on the data and problem type:

- **Convolutional Neural Networks (CNNs)** – often used in computer vision.
- **Recurrent Neural Networks (RNNs)** – useful for sequential data like speech and text.
- **Fully connected (dense) networks** – common for structured/tabular data.
- **Hybrid models** – combining different architectures.

- **Structured data**: organized in tables or databases.  
- **Unstructured data**: images, video, audio, text.
  
 Examples:
| Input (A) | Output (B) | Application |
| :--- | :--- | :--- |
| Image of a product on an assembly line | "Defective" (1) / "OK" (0) | Quality Control |
| A customer's usage data | "Will churn" (1) / "Won't churn" (0) | Customer Churn Prediction |
| Audio file from a call center | Text transcript | Voice Analytics |
| Features of a property | Price (e.g., $350,000) | Real Estate Pricing |

The algorithm's job is to learn the function `f(A) = B` on its own by analyzing thousands or millions of these examples.

---

## 3. Why the Deep Learning "Boom" Now?

Neural networks are not a new idea (they've been around since the 1950s). They became the dominant technology in the last decade due to a "perfect storm" of three factors:

**Analogy: Building a Race Car**

* **Data:** This is the **fuel**. In the past, we had only a gallon of fuel. Today, with Big Data, we have access to a massive volume of data to "feed" our networks. Neural networks, especially large (deep) ones, see their performance improve dramatically as they receive more data.

* **Computation:** These are the **tools and the factory**. Training neural networks requires immense computational power. The development of GPUs (Graphics Processing Units), which can perform many calculations in parallel, gave us the modern, ultra-fast "factory" needed to build complex engines (networks) in a timely manner.

* **Algorithms:** This is the **engine blueprint**. Researchers have made small but incredibly impactful improvements to the algorithms. A simple example was the switch from the Sigmoid activation function to ReLU, which allowed networks to train much faster and avoid certain mathematical problems.

The combination of better engine blueprints, nearly infinite fuel, and high-speed factories enabled the revolution we see today.


---

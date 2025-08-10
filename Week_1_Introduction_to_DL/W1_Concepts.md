# Week 1: An Introduction to Deep Learning

This first section covers the fundamental concepts from Week 1 of Andrew Ng's course. The goal here is to build the intuition behind what neural networks are and why they have become such a transformative force in technology.

## 1. What is a Neural Network?

At its simplest, a neural network is a computational method, inspired by the human brain, designed to recognize patterns. Its primary goal is to learn a complex mathematical function that maps a set of inputs to a desired set of outputs.

Consider predicting a house's price. A simple Linear Regression might use a single feature (like square footage) to draw a straight line. But reality is far more complex. The price depends on the size, number of rooms, location, age of the property, quality of the finishings, and so on. A neural network is capable of "learning" this complex, non-linear relationship.

**Analogy: The Committee of Experts**

We can visualize a neural network as a committee of specialists making a decision:

* **Input Layer:** This is the data-gathering team. Each "neuron" here holds a single piece of information (one knows the house's size, another the number of rooms, etc.). They don't make decisions; they just pass the raw data forward.

* **Hidden Layers:** These are the analysis committees. The first hidden layer receives the raw data and starts finding simple patterns. One neuron might become an expert in the "size vs. number of rooms" relationship to create the concept of "internal space." Another might analyze "location vs. age" to create the concept of "appreciation potential." Subsequent layers are senior expert committees that receive insights from the junior committees and combine them to find even more abstract and complex patterns.

* **Output Layer:** This is the final decision-maker (the CEO). It receives the final recommendation from the senior committees and produces the final output, such as the estimated price of the house.

The process of **training** the network is simply adjusting the "influence" and "confidence" of each expert in every committee until the CEO's final decision is as accurate as possible.

## 2. Why the Deep Learning "Boom" Now?

Neural networks are not a new idea (they've been around since the 1950s). They became the dominant technology in the last decade due to a "perfect storm" of three factors:

**Analogy: Building a Race Car**

* **Data:** This is the **fuel**. In the past, we had only a gallon of fuel. Today, with Big Data, we have access to a massive volume of data to "feed" our networks. Neural networks, especially large (deep) ones, see their performance improve dramatically as they receive more data.

* **Computation:** These are the **tools and the factory**. Training neural networks requires immense computational power. The development of GPUs (Graphics Processing Units), which can perform many calculations in parallel, gave us the modern, ultra-fast "factory" needed to build complex engines (networks) in a timely manner.

* **Algorithms:** This is the **engine blueprint**. Researchers have made small but incredibly impactful improvements to the algorithms. A simple example was the switch from the Sigmoid activation function to ReLU, which allowed networks to train much faster and avoid certain mathematical problems.

The combination of better engine blueprints, nearly infinite fuel, and high-speed factories enabled the revolution we see today.

## 3. Supervised Learning

Most of the successful Deep Learning applications you see today fall under the category of **Supervised Learning**.

The concept is simple: we teach the model by presenting it with a large dataset where we already know the "correct answer." The model learns to map the input to the correct output.

Examples:

| Input (A) | Output (B) | Application |
| :--- | :--- | :--- |
| Image of a product on an assembly line | "Defective" (1) / "OK" (0) | Quality Control |
| A customer's usage data | "Will churn" (1) / "Won't churn" (0) | Customer Churn Prediction |
| Audio file from a call center | Text transcript | Voice Analytics |
| Features of a property | Price (e.g., $350,000) | Real Estate Pricing |

The algorithm's job is to learn the function `f(A) = B` on its own by analyzing thousands or millions of these examples.

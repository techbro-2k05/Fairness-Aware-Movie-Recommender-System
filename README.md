# Fairness-Aware Movie Recommendation System

A recommendation engine framework designed to mitigate algorithmic bias and promote demographic fairness in movie recommendations using Neural Collaborative Filtering (NCF) and decomposed adversarial learning.

## Project Contributors

* **Lakshya Singhal** (23075043)


* **Sairaj Chavan** (23075070)


* **Shanawaz** (23075076)



## Overview

Traditional collaborative filtering systems often inherit and amplify behavioral biases tied to sensitive user attributes such as gender, leading to skewed recommendations. For instance, action movies may be predominantly recommended to specific demographic groups.

This project implements a fairness-aware recommendation framework that decouples user representations into two components:

* **Bias-Aware User Embeddings:** Captures patterns driven by sensitive attributes to model existing biases effectively.


* **Bias-Free User Embeddings:** Encodes attribute-independent features used directly for fair and diverse downstream movie recommendations.



An orthogonality regularization term alongside an adversarial discriminator and predictor setup ensures the clear separation of these biased and fair representations.

## Architecture and Core Components

* **Neural Collaborative Filtering (NeuCF):** Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptrons (MLP) to learn complex user-item interaction dynamics beyond linear inner products.


* **Attribute Predictor:** A neural network operating on bias-aware embeddings to predict sensitive demographic features with minimized loss.


* **Attribute Discriminator:** Adversarially trained on bias-free embeddings to force the network to eliminate sensitive information.


* **Orthogonality Loss:** Minimizes the Mean Squared Error (MSE) between the biased and fair embedding spaces to keep them distinct.



## Evaluation and Metrics

The model is evaluated using training loss convergence and specialized fairness metrics on the MovieLens (`u.data` and `u.user`) dataset:

* Unfairness ($U_{val}$)


* Absolute Unfairness ($U_{abs}$)


* Overestimation Unfairness ($U_{over}$)


* Underestimation Unfairness ($U_{under}$)



## Tech Stack

* **Language:** Python


* **Deep Learning Framework:** PyTorch / TensorFlow
* **Data Processing:** NumPy, Pandas
* **Dataset:** MovieLens (`u.data` for ratings and timestamps; `u.user` for user sensitive attributes like gender)



## Getting Started

### Prerequisites

Ensure Python is installed along with the required libraries:

```bash
pip install numpy pandas torch scikit-learn matplotlib

```

### Dataset Setup

Download the MovieLens dataset files (`u.data` and `u.user`) and place them in the project's data directory.

## Results Summary

* **Binary Crossentropy Loss (Predictor):** Converges rapidly toward near-zero error across training epochs.


* **Fairness Score Evaluation:** Achieves high demographic parity, bringing the discriminator prediction score close to the optimal unbiased threshold ($\sim 0.5$).

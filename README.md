# Why-So-Harsh?

This project focuses on the classification of harsh comments using machine learning techniques. The goal is to develop models that can accurately identify and categorize offensive or inappropriate comments from a given dataset.

## Table of Contents

- [Introduction](#introduction)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Text Pre-processing](#text-pre-processing)
- [Modeling](#modeling)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Introduction

The project is based on a Kaggle competition focused on identifying and classifying harsh comments. The dataset provided consists of comments labeled as either `harsh`, `extremely_harsh`, `vulgar`, `threatening`, `disrespect`, `targeted_hate`. The objective is to develop models that can accurately predict the label for new comments.

## Exploratory Data Analysis

Before proceeding with the modeling, an exploratory data analysis (EDA) was performed on the dataset. This involved analyzing the distribution of comment labels, identifying any class imbalance, and gaining insights into the data characteristics. The EDA helped in understanding the dataset and making informed decisions during the modeling process.

## Text Pre-processing

Text pre-processing is a crucial step in natural language processing tasks. In this project, several pre-processing techniques were applied to clean and normalize the text data. These techniques included:

- Expanding and cleaning up text contractions to ensure consistency.
- Handling and expanding emoticons to capture their meaning.
- Removing stopwords to eliminate common words that do not contribute much to the classification task.

## Modeling

The pre-processed data was then vectorized using both word-level and character-level vectorizers. These vectorization techniques transformed the text data into numerical representations suitable for machine learning models. The following models were trained and evaluated:

- Logistic Regression: A probabilistic model used for binary classification tasks.
- Naive Bayes: A probabilistic model based on Bayes' theorem.
- Random Forest: A tree-based ensemble model that combines multiple decision trees.
- XG Boosting: An ensemble model that sequentially trains weak models and combines their predictions.
- Ridge: A linear model regularized with L2 penalty.
- SGD Classifier: A linear model trained using stochastic gradient descent.

The models were evaluated using validation data, and hyperparameter tuning was performed using techniques like grid search cross-validation for logistic regression.

## Results

Among the models evaluated, logistic regression achieved the best validation score of 0.98587. This indicates that the model performed well in classifying harsh comments based on the given features. Furthermore, the model was tested on the Kaggle dataset, achieving a Kaggle score of 0.98489 with all the available data.

## Dependencies

The following dependencies are required to run the project:

- Python (version X.X.X)
- NumPy (version X.X.X)
- Pandas (version X.X.X)
- Scikit-learn (version X.X.X)
- XGBoost (version X.X.X)

Install the dependencies using the package manager of your choice (e.g., pip, conda).

Please refer to the project documentation for more details and examples.

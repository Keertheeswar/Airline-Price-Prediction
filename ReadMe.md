# Flight Price Prediction

A machine learning project to predict flight prices based on various features and attributes of flight data.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Preprocessing](#preprocessing)
- [Data Exploration](#data-exploration)
- [Model Building](#model-building)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

This project focuses on predicting flight prices, which can be useful for travelers to estimate the cost of their flights. We leverage machine learning techniques to build a predictive model that estimates flight prices based on various attributes such as airline, source, destination, departure time, and more.

## Dataset

The dataset used for this project is available in the file "PA PROJECT DATASET.csv." It contains information about flights, including features like airline, source, destination, departure time, arrival time, and the target variable, which is the flight price.

## Getting Started

To run this project on your local machine, follow these steps:

1. Clone this repository.
2. Make sure you have Python 3.x installed.
3. Install the required libraries listed in the "requirements.txt" file.
4. Replace the file path in the code with the correct path to your dataset.
5. Execute the code and train the machine learning models.

## Preprocessing

- Data is cleaned by removing rows with missing values.
- Date and time columns are converted to datetime objects.
- Features such as the duration of the flight are extracted from the data.
- Categorical variables are one-hot encoded.

## Data Exploration

- Data is explored to understand distributions and relationships between variables.
- Outliers in the target variable (flight prices) are identified and handled.

## Model Building

- Several regression models are trained and evaluated.
- Models include Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Regression.

## Results

- The best-performing model is Random Forest, which achieved the highest accuracy. Here's more about Random Forest:

  ## Random Forest Algorithm

  Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It achieves high accuracy by reducing overfitting through bootstrapping and random feature selection. The ensemble nature of Random Forest captures complex relationships in the data and provides robust predictions.

- The performance of each model is evaluated using metrics like R-squared, Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error.

## Usage

You can use this project as a basis for predicting flight prices or as an example of a machine learning regression task. Feel free to customize it and use your dataset for similar prediction tasks.

## Contributing

Contributions to this project are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

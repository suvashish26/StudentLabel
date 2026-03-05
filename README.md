# Student Pass/Fail Prediction using Random Forest

## Project Overview

This project uses a simple machine learning model to predict whether a
student will **Pass** or **Fail** based on study and lifestyle habits.

The model is implemented in **Python** using **NumPy** and
**scikit-learn**. It demonstrates a basic supervised learning
classification workflow.

------------------------------------------------------------------------

## Features Used

The model uses the following input features:

-   Study Hours -- Hours spent studying per day
-   Sleep Hours -- Average sleep hours per night
-   Attendance -- Attendance percentage
-   Previous Grades -- Previous academic score out of 100
-   Homework Hours -- Time spent doing homework

These values are stored in a NumPy array called **X**.

------------------------------------------------------------------------

## Target Variable

The target variable is stored in **Y** and represents:

-   Pass
-   Fail

The model learns patterns from the training data to predict these
outcomes.

------------------------------------------------------------------------

## Machine Learning Model

The project uses the **RandomForestClassifier** from scikit-learn.

Random Forest is an **ensemble learning algorithm** that: - Builds
multiple decision trees - Combines their predictions - Produces a more
stable and accurate result

In this project:

n_estimators = 10

This means the model builds **10 decision trees**.

------------------------------------------------------------------------

## Program Workflow

### 1. Import Libraries

``` python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
```

### 2. Define Dataset

The training dataset contains student behavior data and outcomes.

-   X → input features
-   Y → pass/fail labels

### 3. Train the Model

``` python
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
```

The `.fit()` function trains the model.

### 4. Make Prediction

``` python
prediction = clf.predict([[8,1,90,90,2]])
```

This predicts whether a student with these attributes will pass or fail.

------------------------------------------------------------------------

## Requirements

Install required libraries:

    pip install numpy scikit-learn

------------------------------------------------------------------------

## How to Run

1.  Save the Python script
2.  Run:


    python student_prediction.py

------------------------------------------------------------------------

## Learning Objectives

This project demonstrates:

-   Basic machine learning workflow
-   Supervised learning
-   Classification models
-   Training models with `.fit()`
-   Making predictions with `.predict()`
-   Using NumPy arrays for datasets

------------------------------------------------------------------------

## Possible Improvements

-   Add a larger dataset
-   Evaluate model accuracy
-   Add confusion matrix
-   Add visualization of feature importance

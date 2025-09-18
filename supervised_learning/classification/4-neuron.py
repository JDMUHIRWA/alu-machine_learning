#!/usr/bin/env python3
"""Defines a single neuron performing binary classification."""
import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initialize the neuron.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        # Validate nx type
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # Validate nx value
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize public instance attributes
        self.__W = np.random.randn(1, nx)  # Random normal distribution
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activated output initialized to 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Perform forward propagation.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: Activated output (prediction).
        """
        Z = np.dot(self.__W, X) + self.__b  # Weighted sum
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def cost(self, Y, A):
        """
        Calculating the cost of the model using logistic
        regression

        Args:
            Y(numpy.ndarray): Actual labels for the input data
            A(numpy.ndarray): activated output of the neuron for each example

        Returns:
            float: Returns the cost of the model
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Actual labels of shape (1, m).

        Returns:
            tuple: The neuron’s prediction and the cost of the model.
        """
        A = self.forward_prop(X)  # Forward propagation
        cost = self.cost(Y, A)  # Calculating the cost
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

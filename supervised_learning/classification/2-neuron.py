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

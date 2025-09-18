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
        self.W = np.random.randn(1, nx)  # Random normal distribution
        self.b = 0  # Bias initialized to 0
        self.A = 0  # Activated output initialized to 0

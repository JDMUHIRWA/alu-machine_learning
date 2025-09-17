#!/usr/bin/env python3
import numpy as np  # Required for random normal distribution

class Neuron:
    # Defining the constructor
    def __init__(self, nx):
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

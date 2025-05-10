#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here
plt.scatter(x,y)
plt.title('Men\'s Height vs Height')
plt.xlabel('Weight (in)')
plt.ylabel('Height (lbs)')
plt.legend(['Men\'s Height vs Weight'])
plt.show()


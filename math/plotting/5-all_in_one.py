#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

#Creating Figure
fig,axs = plt.subplots(3,2,figsize=(12,12))

#line graph
axs[0,0].plot(y0, color='red')

#scatter graph
axs[0,1].scatter(x1,y1, color='magenta')
axs[0,1].set_title("Men's Height vs Weight", fontsize='x-small')
axs[0,1].set_xlabel('Height(in)', fontsize='x-small')
axs[0,1].set_ylabel('Weight(lbs)',fontsize='x-small')

#change_scale
axs[1,0].plot(x2,y2, color= 'blue')
axs[1,0].set_title('Exponential Decay of C-14', fontsize='x-small')
axs[1,0].set_xlabel('Time(years)', fontsize='x-small')
axs[1,0].set_ylabel('Fraction Remaining', fontsize='x-small')

#twoo
axs[1,0].plot(x3,y31, 'r--')



plt.show()
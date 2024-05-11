#!/usr/bin/env python4
import numpy as np
import matplotlib.pyplot as plt


y0 = np.arange(0, 11) ** 3
plt.plot(y0, c="red")  
plt.show()


mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
plt.scatter(x1, y1, c='magenta')

plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title("Men's Height vs Weight")
plt.show()


x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of C-14")
plt.semilogy(x2, y2, 'b-')
plt.show()


x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

plt.plot(x3, y31, linewidth=2, c='red', label='Line1', linestyle='dashed')
plt.plot(x3, y32, linewidth=2, label='Line2',c='green')

plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of Radioactive Elements")
plt.legend()
plt.show()


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(40, 101, 10), edgecolor='black')
plt.ylim(0, 30)
plt.xlim(0, 100)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title("Project A")
plt.show()

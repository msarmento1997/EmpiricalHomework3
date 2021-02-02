# Michael Sarmento
# CIS 362 HW3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.stats as sc
from prettytable import PrettyTable


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv("airfoil_self_noise.csv")

print('\n', data.describe(), '\n\n')

# print correlation
print('Correlation Table:\n', data.corr(), '\n\n')

# angle of attack and suction side displacement (.7533)
# chord length and angle of attack (-.5048)

# creating data frames for each data category
mylist1 = data["Frequency"].tolist()
mylist2 = data["Angle of attack"].tolist()
mylist3 = data["Chord length"].tolist()
mylist4 = data["Free-stream velocity"].tolist()
mylist5 = data["Suction side displacement"].tolist()
mylist6 = data["Scaled sound pressure level"].tolist()


x = mylist2
y = mylist5

x2 = mylist2
y2 = mylist3

# used from example document provided for regression
slope1, intercept1, rvalue1, pvalue1, stderr1 = sc.linregress(x, y)
fit1 = [slope1, intercept1]

# used from example document provided for regression
slope2, intercept2, rvalue2, pvalue2, stderr2 = sc.linregress(x2, y2)
fit2 = [slope2, intercept2]

# table displaying regression information between two categories
t1 = PrettyTable(['Suction Side Displacement', 'Angle of Attack'])
t1.add_row(['Slope', slope1])
t1.add_row(['Y-intercept', intercept1])
t1.add_row(['R value', rvalue1])
t1.add_row(['R-squared', pow(rvalue1, 2)])
t1.add_row(['P value', pvalue1])
t1.add_row(['Std err', stderr1])
print(t1)
print('Regression Model: Suction Side Displacement = ', intercept1, ' + ', slope1, ' * Angle_of_Attack(in degrees)\n\n')

# table displaying regression information between two categories
t2 = PrettyTable(['Chord Length', 'Angle of Attack'])
t2.add_row(['Slope', slope2])
t2.add_row(['Y-intercept', intercept2])
t2.add_row(['R value', rvalue2])
t2.add_row(['R-squared', pow(rvalue2, 2)])
t2.add_row(['P value', pvalue2])
t2.add_row(['Std err', stderr2])
print(t2)
print('Regression Model: Chord Length = ', intercept2, ' + ', slope2, ' * Angle_of_Attack(in degrees)\n\n')


# plot given values and regression prediction line
plt.plot(x, y, 'x')
plt.plot(x, np.polyval(fit1, x), 'r-')
plt.xlabel('Angle of Attack')
plt.ylabel('Suction Side Displacement')
plt.title('Simple Linear Regression (Angle of Attack vs Suction Side Displacement)')
plt.show()

# plot given values and regression prediction line
plt.plot(x2, y2, 'x')
plt.plot(x2, np.polyval(fit2, x2), 'r-')
plt.xlabel('Angle of Attack')
plt.ylabel('Chord Length')
plt.title('Simple Linear Regression (Angle of Attack vs Chord length)')
plt.show()


# box plot for frequency
plt.boxplot(mylist1)
plt.title("Frequency in Hertz")
plt.xlabel("Frequency")
plt.show()

# box plot for Angle of attack
plt.boxplot(mylist2)
plt.title("Angle of Attack in Degrees")
plt.xlabel("Angle of Attack")
plt.show()

# box plot for Chord length
plt.boxplot(mylist3)
plt.title("Chord Length in Meters")
plt.xlabel("Chord Length")
plt.show()

# box plot for Free-stream velocity
plt.boxplot(mylist4)
plt.title("Free-Stream Velocity, in Meters per Second")
plt.xlabel("Free-Stream Velocity")
plt.show()

# box plot for Free-stream velocity
plt.boxplot(mylist5)
plt.title("Suction Side Displacement Thickness, in Meters")
plt.xlabel("Suction Side Displacement")
plt.show()

# box plot for Scaled Sound pressure level
plt.boxplot(mylist6)
plt.title("Scaled Sound Pressure Level, in Decibels")
plt.xlabel("Scaled Sound Pressure Level")
plt.show()

# box plot for frequency
plt.hist(mylist1)
plt.title("Frequency in Hertz")
plt.xlabel("Frequency")
plt.show()

# box plot for Angle of attack
plt.hist(mylist2)
plt.title("Angle of Attack in Degrees")
plt.xlabel("Angle of Attack")
plt.show()

# box plot for Chord length
plt.hist(mylist3)
plt.title("Chord Length in Meters")
plt.xlabel("Chord Length")
plt.show()

# box plot for Free-stream velocity
plt.hist(mylist4)
plt.title("Free-Stream Velocity, in Meters per Second")
plt.xlabel("Free-Stream Velocity")
plt.show()

# box plot for Free-stream velocity
plt.hist(mylist5)
plt.title("Suction Side Displacement Thickness, in Meters")
plt.xlabel("Suction Side Displacement")
plt.show()

# box plot for F
plt.hist(mylist6)
plt.title("Scaled Sound Pressure Level, in Decibels")
plt.xlabel("Scaled Sound Pressure Level")
plt.show()

# correlation heat map
sn.heatmap(data.corr(), annot=True)
plt.show()



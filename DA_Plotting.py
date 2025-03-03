import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import GK_Data_Analysis as DA

my_path = os.path.dirname(os.path.abspath(__file__))

# Plot the data

#Torque vs RPM for each applied voltage (all plots on a single graph)
sns.set()
fig, ax = plt.subplots()
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], label=f'{voltage} V')
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N-m)')
ax.legend(title='Applied Voltage')
plt.title('Torque vs RPM for each applied voltage')
plt.savefig(os.path.join(my_path, 'Plots', 'Torque_vs_RPM.png'))
plt.show()

#plot motor speed vs mechanical load (N) for all voltages in one plot
fig, ax = plt.subplots()
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Scale Reading (g)']*0.00981, df_v['Tach Reading (RPM)'], label=f'{voltage} V')
ax.set_xlabel('Mechanical Load (N)')
ax.set_ylabel('RPM')
ax.legend(title='Applied Voltage')
plt.title('Angular Speed (RPM) vs Mechanical Load (N) for each applied voltage')
plt.savefig(os.path.join(my_path, 'Plots/Angular_Speed_vs_Mechanical_Load.png'))
plt.show()

#plot current draw vs mechanical load (N) for all voltages in one plot
fig, ax = plt.subplots()
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Scale Reading (g)']*0.00981, df_v['Current Draw (A)'], label=f'{voltage} V')
ax.set_xlabel('Mechanical Load (N)')
ax.set_ylabel('Current Draw (A)')
ax.legend(title='Applied Voltage')
plt.title('Current Draw (A) vs Mechanical Load (N) for each applied voltage')
plt.savefig(os.path.join(my_path, 'Plots/Current_Draw_vs_Mechanical_Load.png'))
plt.show()

#plot mechanical power vs efficiency for each applied voltage
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    df_v = df_v.sort_values(by='Torque (N-m)')
    coefficients = np.polyfit(df_v['Torque (N-m)'], df_v['Efficiency'], 2)
    polynomial = np.poly1d(coefficients)
    torque_range = np.linspace(df_v['Torque (N-m)'].min(), df_v['Torque (N-m)'].max(), 100)
    plt.plot(torque_range, polynomial(torque_range), label=f'Fit: {coefficients[0]:.6f}x^2 + {coefficients[1]:.4f}x+{coefficients[2]:.4f}')
    ax.scatter(df_v['Torque (N-m)'], df_v['Efficiency'])
    plt.legend()
    ax.set_xlabel('Torque (N-m)')
    ax.set_ylabel('Efficiency')
    plt.title(f'Mechanical Power vs Efficiency for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/MechPwr_Vs_Efficiency/Mechanical_Power_vs_Efficiency_{voltage}V.png'))
plt.show()

#Seperate plots of torque vs speed for each voltage level
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    coefficients = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    polynomial = np.poly1d(coefficients)
    #add horizontal and vertical error bars
    ax.errorbar(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], xerr=0.05, yerr=abs(df_v['Relative Error Torque']), fmt='o')
    
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'])
    plt.plot(df_v['Tach Reading (RPM)'], polynomial(df_v['Tach Reading (RPM)']), label=f'Fit: {coefficients[0]:.6f}x + {coefficients[1]:.4f}')
    plt.legend()
    ax.set_xlabel('RPM')
    ax.set_ylabel('Torque (N-m)')
    plt.title(f'Torque vs RPM for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Torque_vs_RPM/Torque_vs_RPM_{voltage}V.png'))
plt.show()



#Plot torque vs current load for each applied voltage
torque_constant = []
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    df_v = df_v.sort_values(by='Current Draw (A)')
    coefficients = np.polyfit(df_v['Current Draw (A)'], df_v['Torque (N-m)'], 1)
    polynomial = np.poly1d(coefficients)
    #the slope of the line is the motor torque constant
    #the average slope
    torque_constant.append(coefficients[0])
    plt.plot(df_v['Current Draw (A)'], polynomial(df_v['Current Draw (A)']), label=f'Fit: {coefficients[0]:.6f}x + {coefficients[1]:.4f}')
    ax.scatter(df_v['Current Draw (A)'], df_v['Torque (N-m)'])
    plt.legend()
    ax.set_xlabel('Current Draw (A)')
    ax.set_ylabel('Torque (N-m)')
    plt.title(f'Torque vs Current Draw for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Torque_vs_Current/Torque_vs_Current_{voltage}V.png'))
plt.show()

#Plot the speed vs current load curve for each applied voltage
K_e = [] #Back EMF constant
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    df_v = df_v.sort_values(by='Current Draw (A)')
    coefficients = np.polyfit(df_v['Current Draw (A)'], df_v['Tach Reading (RPM)'], 1)
    polynomial = np.poly1d(coefficients)
    #calculate K_e for each voltage level and append to list
    K_e.append(-voltage/(coefficients[0]* (2 * np.pi / 60)))
    plt.plot(df_v['Current Draw (A)'], polynomial(df_v['Current Draw (A)']), label=f'Fit: {coefficients[0]:.6f}x + {coefficients[1]:.4f}')
    ax.scatter(df_v['Current Draw (A)'], df_v['Tach Reading (RPM)'])
    plt.legend()
    ax.set_xlabel('Current Draw (A)')
    ax.set_ylabel('RPM')
    plt.title(f'RPM vs Current Draw for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/RPM_vs_Current/RPM_vs_Current_{voltage}V.png'))
plt.show()




#add motor properties to text file and print the file
with open('Motor_Properties.txt', 'w') as f:
    f.write("Properties of the motor:\n")
    f.write("The mean Motor Torque Constant is: " + str(np.mean(torque_constant)) + " N-m/A\n")
    f.write("The peak efficiency is " + str(DA.df['Efficiency'].max()) + " at a voltage of " + str(DA.df['Applied Voltage (v)'][DA.df['Efficiency'].idxmax()]) + "\n")
    f.write("The mean Back EMF constant is: " + str(np.mean(K_e)) + " V-s/rad\n")
    f.write("The average resistance is: " + str(DA.average_resistance) + " Ohms\n")

with open('Motor_Properties.txt', 'r') as f:
    print(f.read())


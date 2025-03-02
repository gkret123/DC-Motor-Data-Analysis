import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file into a df
file_path = "GK_Data_M1.xlsx"
df = pd.read_excel(file_path)

# Display the first few rows 
print("First 5 rows:")
print(df.head())
print("\nDataframe info:")
print(df.info())

# Check for missing values per column
print("\nMissing values per column:")
print(df.isnull().sum())

if 'Torque (N-m)' in df.columns:
    df['Torque (N-m)'] = df.apply(
        lambda row: 0.0981 * (row['Scale Reading (g)']- row['Lever arm load  (end) (g)'] )* 0.001 * row['Moment arm (cm)'], axis=1
    )
if 'Electrical Power (W)' in df.columns:
    df['Electrical Power (W)'] = df.apply(
        lambda row: row['Applied Voltage (v)'] * row['Current Draw (A)'], axis=1
    )
if 'Mechanical Power (W)' in df.columns:
    df['Mechanical Power (W)'] = df.apply(
        lambda row: row['Torque (N-m)'] * row['Tach Reading (RPM)'] * 2 * np.pi/60 , axis=1
    )
if 'Efficiency' in df.columns:
    df['Efficiency'] = df.apply(
        lambda row: row['Mechanical Power (W)'] / row['Electrical Power (W)'], axis=1
    )
if 'Relative Error Electrical Power' in df.columns:
    df['Relative Error Electrical Power'] = df.apply(
        lambda row: row['Electrical Power (W)']*np.sqrt((0.005/row['Applied Voltage (v)'])**2 + (0.005/row['Current Draw (A)'])**2), axis=1
    )
if 'Relative Error Torque' in df.columns:
    df['Relative Error Torque'] = df.apply(
        lambda row: row['Torque (N-m)']*np.sqrt((0.005/row['Moment arm (cm)'])**2 + (0.05/row['Scale Reading (g)'])**2), axis=1
    )
#add Resistance column
df['Resistance (Ohms)'] = df['Applied Voltage (v)']/df['Current Draw (A)']
average_resistance = df['Resistance (Ohms)'].mean()
# Verify the missing values got filled
print("\nMissing values per column after filling:")
print(df.isnull().sum())

#save to new excel file
df.to_excel("GK_Data_M1_filled.xlsx", index=False)



# Plot the data

#Torque vs RPM for each applied voltage
sns.set()
fig, ax = plt.subplots()
for voltage in df['Applied Voltage (v)'].unique():
    df_v = df[df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], label=f'{voltage} V')
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N-m)')
ax.legend(title='Applied Voltage')
plt.title('Torque vs RPM for each applied voltage')
plt.show()

#Seperate plots for each voltage level
for voltage in df['Applied Voltage (v)'].unique():
    df_v = df[df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    coefficients = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    polynomial = np.poly1d(coefficients)
    #add horizontal and vertical error bars
    ax.errorbar(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], xerr=abs(df_v['Relative Error Torque']), yerr=0.05, fmt='o')
    
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'])
    plt.plot(df_v['Tach Reading (RPM)'], polynomial(df_v['Tach Reading (RPM)']), label=f'Fit: {coefficients[0]:.6f}x + {coefficients[1]:.4f}')
    plt.legend()
    ax.set_xlabel('RPM')
    ax.set_ylabel('Torque (N-m)')
    plt.title(f'Torque vs RPM for {voltage} V')
    plt.show()

#plot mechanical power vs efficiency for each applied voltage
for voltage in df['Applied Voltage (v)'].unique():
    df_v = df[df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    df_v = df_v.sort_values(by='Mechanical Power (W)')
    coefficients = np.polyfit(df_v['Mechanical Power (W)'], df_v['Efficiency'], 2)
    polynomial = np.poly1d(coefficients)
    plt.plot(df_v['Mechanical Power (W)'], polynomial(df_v['Mechanical Power (W)']), label=f'Fit: {coefficients[0]:.6f}x^2 + {coefficients[1]:.4f}x+{coefficients[2]:.4f}')
    ax.scatter(sorted(df_v['Mechanical Power (W)']), df_v['Efficiency'])
    ax.set_xlabel('Mechanical Power (W)')
    ax.set_ylabel('Efficiency')
    plt.title(f'Mechanical Power vs Efficiency for {voltage} V')
    plt.show()

#Plot torque vs current load for each applied voltage
torque_constant = []
for voltage in df['Applied Voltage (v)'].unique():
    df_v = df[df['Applied Voltage (v)'] == voltage]
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
    plt.show()
#Plot the speed vs current load curve for each applied voltage
K_e = [] #Back EMF constant
for voltage in df['Applied Voltage (v)'].unique():
    df_v = df[df['Applied Voltage (v)'] == voltage]
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
    plt.show()

#Print the properties of the motor
print("Properties of the motor:")
print("The mean Motor Torque Constant is: ", np.mean(torque_constant), "N-m/A")
print("The peak efficiency is ", df['Efficiency'].max() , "at a voltage of ", df['Applied Voltage (v)'][df['Efficiency'].idxmax()])
print("The mean Back EMF constant is: ", np.mean(K_e), "V-s/rad")
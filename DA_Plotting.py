import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import GK_Data_Analysis as DA

my_path = os.path.dirname(os.path.abspath(__file__))
sns.set()

# Plot Mechanical Power vs RPM for each applied voltage
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
    fig, ax = plt.subplots()
    coeffs_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    coeffs_Current_vs_RPM = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Current Draw (A)'], 1)
    stall_torque = coeffs_RPM_vs_Torque[1]
    
    print(f"Voltage: {voltage}")
    print(f"Stall Torque: {stall_torque}")
    print(f"RPM vs Torque Coefficients: {coeffs_RPM_vs_Torque}")
    print(f"Current vs RPM Coefficients: {coeffs_Current_vs_RPM}")

    torque_at_rpm = stall_torque + coeffs_RPM_vs_Torque[0] * df_v['Tach Reading (RPM)']
    current_at_rpm = coeffs_Current_vs_RPM[0] * df_v['Tach Reading (RPM)'] + coeffs_Current_vs_RPM[1]
    power_output = torque_at_rpm * df_v['Tach Reading (RPM)'] * (2 * np.pi / 60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input

    ax.scatter(df_v['Tach Reading (RPM)'], power_output, label="Mechanical Power Output")
    max_power_index = np.argmax(power_output)
    ax.text(df_v['Tach Reading (RPM)'].iloc[max_power_index] + df_v['Tach Reading (RPM)'].iloc[max_power_index] / 2,
            power_output.iloc[max_power_index],
            f"Max Power: {power_output.iloc[max_power_index]:.2f} W \n at {df_v['Tach Reading (RPM)'].iloc[max_power_index]} RPM",
            fontsize=11)

    plt.legend()
    ax.set_xlabel('RPM')
    ax.set_ylabel('Mechanical Power (W)')
    plt.title(f'Power vs RPM for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Power_vs_RPM/Power_vs_RPM_{voltage}V.png'))
#plt.show()

voltages = [5.0, 9.53, 11.0]
fig, ax = plt.subplots(figsize=(20, 12))
for voltage in voltages:
    df_v_3 = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
    coeffs_RPM_vs_Torque = np.polyfit(df_v_3['Tach Reading (RPM)'], df_v_3['Torque (N-m)'], 1)
    coeffs_Current_vs_RPM = np.polyfit(df_v_3['Tach Reading (RPM)'], df_v_3['Current Draw (A)'], 1)
    stall_torque = coeffs_RPM_vs_Torque[1]
    
    torque_at_rpm = stall_torque + coeffs_RPM_vs_Torque[0] * df_v_3['Tach Reading (RPM)']
    current_at_rpm = coeffs_Current_vs_RPM[0] * df_v_3['Tach Reading (RPM)'] + coeffs_Current_vs_RPM[1]
    power_output = torque_at_rpm * df_v_3['Tach Reading (RPM)'] * (2 * np.pi / 60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input
    
    ax.scatter(df_v_3['Tach Reading (RPM)'], power_output, label="Mechanical Power Output")

    max_power_index = np.argmax(power_output)

    #add max power point and its corresponding torque and RPM to the legend
    ax.scatter(df_v_3['Tach Reading (RPM)'].iloc[max_power_index], power_output.iloc[max_power_index],
               label=f'Max Power: {power_output.iloc[max_power_index]:.2f} W at {df_v_3["Tach Reading (RPM)"].iloc[max_power_index]} RPM for {voltage} V')
ax.set_xlabel('RPM')
ax.set_ylabel('Mechanical Power Output (W)')
ax.legend(title='Applied Voltage')
plt.title('Mechanical Power Output vs RPM for 5V, 9.53V, and 11V')
plt.savefig(os.path.join(my_path, 'Plots/Power_vs_RPM/Power_vs_RPM_3_Voltages.png'))
#plt.show()

# Plot Torque vs RPM for each applied voltage (all on a single graph)
fig, ax = plt.subplots()
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], label=f'{voltage} V')
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N-m)')
ax.legend(title='Applied Voltage')
plt.title('Torque vs RPM for each applied voltage')
plt.savefig(os.path.join(my_path, 'Plots', 'Torque_vs_RPM.png'))
#plt.show()

# Plot Motor Speed vs Mechanical Load for all voltages
fig, ax = plt.subplots()
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Scale Reading (g)'] * 0.00981, df_v['Tach Reading (RPM)'], label=f'{voltage} V')
ax.set_xlabel('Mechanical Load (N)')
ax.set_ylabel('RPM')
ax.legend(title='Applied Voltage')
plt.title('Angular Speed (RPM) vs Mechanical Load (N) for each applied voltage')
plt.savefig(os.path.join(my_path, 'Plots/Angular_Speed_vs_Mechanical_Load.png'))
#plt.show()

# Plot Current Draw vs Torque for all voltages
fig, ax = plt.subplots()
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    ax.scatter(df_v['Torque (N-m)'], df_v['Current Draw (A)'], label=f'{voltage} V')
ax.set_xlabel('Torque (N-m)')
ax.set_ylabel('Current Draw (A)')
ax.legend(title='Applied Voltage')
plt.title('Current Draw (A) vs Torque (N-m) for each applied voltage')
plt.savefig(os.path.join(my_path, 'Plots/Current_Draw_vs_Mechanical_Load.png'))
#plt.show()

# Separate plots of Torque vs RPM for each voltage level
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    coeffs_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    poly = np.poly1d(coeffs_RPM_vs_Torque)
    ax.errorbar(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], xerr=0.05,
                yerr=abs(df_v['Relative Error Torque']), fmt='o')
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'])
    plt.plot(df_v['Tach Reading (RPM)'], poly(df_v['Tach Reading (RPM)']),
             label=f'Fit: {coeffs_RPM_vs_Torque[0]:.6f}x + {coeffs_RPM_vs_Torque[1]:.4f}')
    plt.legend()
    ax.set_xlabel('RPM')
    ax.set_ylabel('Torque (N-m)')
    plt.title(f'Torque vs RPM for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Torque_vs_RPM/Torque_vs_RPM_{voltage}V.png'))
#plt.show()

voltages = [5.0, 9.53, 11.0]
fig, ax = plt.subplots(figsize=(20, 12))
for voltage in voltages:
    df_v_3 = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    coeffs_RPM_vs_Torque = np.polyfit(df_v_3['Tach Reading (RPM)'], df_v_3['Torque (N-m)'], 1)
    poly = np.poly1d(coeffs_RPM_vs_Torque)
    plt.plot(df_v_3['Tach Reading (RPM)'], poly(df_v_3['Tach Reading (RPM)']),
             label=f'Fit: {coeffs_RPM_vs_Torque[0]:.6f}x + {coeffs_RPM_vs_Torque[1]:.4f}')
    ax.scatter(df_v_3['Tach Reading (RPM)'], df_v_3['Torque (N-m)'], label=f'{voltage} V')
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N-m)')
ax.legend(title='Applied Voltage')
plt.title('Torque vs RPM for 5V, 9.53V, and 11V')
plt.savefig(os.path.join(my_path, 'Plots/Torque_vs_RPM/Torque_vs_RPM_3_Voltages.png'))
#plt.show()

# Plot Torque vs Current Draw for each voltage and record the torque constant
torque_constant = []
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Current Draw (A)')
    fig, ax = plt.subplots()
    coeffs_Current_vs_Torque = np.polyfit(df_v['Current Draw (A)'], df_v['Torque (N-m)'], 1)
    poly = np.poly1d(coeffs_Current_vs_Torque)
    torque_constant.append(coeffs_Current_vs_Torque[0])
    plt.plot(df_v['Current Draw (A)'], poly(df_v['Current Draw (A)']),
             label=f'Fit: {coeffs_Current_vs_Torque[0]:.6f}x + {coeffs_Current_vs_Torque[1]:.4f}')
    ax.scatter(df_v['Current Draw (A)'], df_v['Torque (N-m)'])
    plt.legend()
    ax.set_xlabel('Current Draw (A)')
    ax.set_ylabel('Torque (N-m)')
    plt.title(f'Torque vs Current Draw for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Torque_vs_Current/Torque_vs_Current_{voltage}V.png'))
#plt.show()

voltages = [5.0, 9.53, 11.0]
fig, ax = plt.subplots(figsize=(20, 12))
for voltage in voltages:
    df_v_3 = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    coeffs_RPM_vs_Torque = np.polyfit(df_v_3['Current Draw (A)'], df_v_3['Torque (N-m)'], 1)
    poly = np.poly1d(coeffs_RPM_vs_Torque)
    plt.plot(df_v_3['Current Draw (A)'], poly(df_v_3['Current Draw (A)']),
             label=f'Fit: {coeffs_RPM_vs_Torque[0]:.6f}x + {coeffs_RPM_vs_Torque[1]:.4f}')
    ax.scatter(df_v_3['Current Draw (A)'], df_v_3['Torque (N-m)'], label=f'{voltage} V')
ax.set_xlabel('Current Draw (A)')
ax.set_ylabel('Torque (N-m)')
ax.legend(title='Applied Voltage')
plt.title('Torque vs Current for 5V, 9.53V, and 11V')
plt.savefig(os.path.join(my_path, 'Plots/Torque_vs_Current/Torque_vs_Current_3_Voltages.png'))
#plt.show()

# Plot RPM vs Current Draw for each voltage and calculate Back EMF constant
K_e = []  # Back EMF constant
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Current Draw (A)')
    fig, ax = plt.subplots()
    coeffs_Current_vs_RPM = np.polyfit(df_v['Current Draw (A)'], df_v['Tach Reading (RPM)'], 1)
    poly = np.poly1d(coeffs_Current_vs_RPM)
    K_e.append(-voltage / (coeffs_Current_vs_RPM[0] * (2 * np.pi / 60)))
    plt.plot(df_v['Current Draw (A)'], poly(df_v['Current Draw (A)']),
             label=f'Fit: {coeffs_Current_vs_RPM[0]:.6f}x + {coeffs_Current_vs_RPM[1]:.4f}')
    ax.scatter(df_v['Current Draw (A)'], df_v['Tach Reading (RPM)'])
    plt.legend()
    ax.set_xlabel('Current Draw (A)')
    ax.set_ylabel('RPM')
    plt.title(f'RPM vs Current Draw for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/RPM_vs_Current/RPM_vs_Current_{voltage}V.png'))
#plt.show()
voltages = [5.0, 9.53, 11.0]
fig, ax = plt.subplots(figsize=(20, 12))
for voltage in voltages:
    df_v_3 = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    coeffs_RPM_vs_Torque = np.polyfit(df_v_3['Current Draw (A)'], df_v_3['Tach Reading (RPM)'], 1)
    poly = np.poly1d(coeffs_RPM_vs_Torque)
    plt.plot(df_v_3['Current Draw (A)'], poly(df_v_3['Current Draw (A)']),
             label=f'Fit: {coeffs_RPM_vs_Torque[0]:.6f}x + {coeffs_RPM_vs_Torque[1]:.4f}')
    ax.scatter(df_v_3['Current Draw (A)'], df_v_3['Tach Reading (RPM)'], label=f'{voltage} V')
ax.set_xlabel('Current Draw (A)')
ax.set_ylabel('Tach Reading (RPM)')
ax.legend(title='Applied Voltage')
plt.title('RPM vs Current for 5V, 9.53V, and 11V')
plt.savefig(os.path.join(my_path, 'Plots/RPM_vs_Current/RPM_vs_Current_3_Voltages.png'))
#plt.show()

# Plot Efficiency vs Current Draw for each voltage
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Current Draw (A)')
    fig, ax = plt.subplots()
    ax.scatter(df_v['Current Draw (A)'], df_v['Efficiency'], label='Efficiency from Data')
    ax.set_xlabel('Current Draw (A)')
    ax.set_ylabel('Efficiency')
    ax.set_title(f'Efficiency vs Current Draw for {voltage} V')
    ax.legend()
    plt.savefig(os.path.join(my_path, f'Plots/Efficiency_vs_Current/Efficiency_vs_Current_{voltage}V.png'))
#plt.show()

# Plot Efficiency vs RPM for each voltage using trendline coefficients
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
    fig, ax = plt.subplots()
    coeffs_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    coeffs_Current_vs_RPM = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Current Draw (A)'], 1)
    stall_torque = coeffs_RPM_vs_Torque[1]

    print(f"Voltage: {voltage}")
    print(f"Stall Torque: {stall_torque}")
    print(f"RPM vs Torque Coefficients: {coeffs_RPM_vs_Torque}")
    print(f"Current vs RPM Coefficients: {coeffs_Current_vs_RPM}")

    torque_at_rpm = stall_torque + coeffs_RPM_vs_Torque[0] * df_v['Tach Reading (RPM)']
    current_at_rpm = coeffs_Current_vs_RPM[0] * df_v['Tach Reading (RPM)'] + coeffs_Current_vs_RPM[1]
    power_output = torque_at_rpm * df_v['Tach Reading (RPM)'] * (2 * np.pi / 60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input
    RPM_range = np.linspace(df_v['Tach Reading (RPM)'].min(), df_v['Tach Reading (RPM)'].max(), len(eff))
    
    ax.scatter(RPM_range, eff,
               label=r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} + B\omega)\omega}{(A\omega + \omega_{0})V}$")
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Efficiency'],
               label=r'Direct from Data: $\eta = \frac{\tau\omega}{I V}$')
    
    #add max efficiency point and its corresponding torque and RPM in the legend
    max_eff_index = df_v['Efficiency'].idxmax()
    max_eff_torque = df_v['Torque (N-m)'][max_eff_index]
    max_eff_rpm = df_v['Tach Reading (RPM)'][max_eff_index]
    ax.scatter(max_eff_rpm, df_v['Efficiency'].max(), label=f'Max Efficiency: {df_v["Efficiency"].max():.2f} at '
                                                             f'{max_eff_torque:.2f} N-m and {max_eff_rpm} RPM')
    plt.legend()
    ax.set_ylim(0, df_v['Efficiency'].max() + 0.1)
    ax.set_xlabel('RPM')
    ax.set_ylabel('Efficiency')
    plt.title(f'Efficiency vs RPM for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Efficiency_vs_RPM/Efficiency_vs_RPM_{voltage}V.png'))
#plt.show()

# Plot Efficiency vs Torque for each voltage using trendline coefficients
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
    fig, ax = plt.subplots()
    coeffs_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    coeffs_Current_vs_RPM = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Current Draw (A)'], 1)
    stall_torque = coeffs_RPM_vs_Torque[1]
    
    print(f"Voltage: {voltage}")
    print(f"Stall Torque: {stall_torque}")
    print(f"RPM vs Torque Coefficients: {coeffs_RPM_vs_Torque}")
    print(f"Current vs RPM Coefficients: {coeffs_Current_vs_RPM}")

    torque_at_rpm = stall_torque + coeffs_RPM_vs_Torque[0] * df_v['Tach Reading (RPM)']
    current_at_rpm = coeffs_Current_vs_RPM[0] * df_v['Tach Reading (RPM)'] + coeffs_Current_vs_RPM[1]
    power_output = torque_at_rpm * df_v['Tach Reading (RPM)'] * (2 * np.pi / 60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input
    RPM_range = np.linspace(df_v['Tach Reading (RPM)'].min(), df_v['Tach Reading (RPM)'].max(), len(eff))
    
    ax.scatter(torque_at_rpm, eff,
               label=r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} + B\omega)\omega}{(A\omega + \omega_{0})V}$")
    ax.scatter(df_v['Torque (N-m)'], df_v['Efficiency'],
               label=r'Direct from Data: $\eta = \frac{\tau\omega}{I V}$')
        #add max efficiency point and its corresponding torque and RPM in the legend
    max_eff_index = df_v['Efficiency'].idxmax()
    max_eff_torque = df_v['Torque (N-m)'][max_eff_index]
    max_eff_rpm = df_v['Tach Reading (RPM)'][max_eff_index]
    ax.scatter(max_eff_torque, df_v['Efficiency'].max(), label=f'Max Efficiency: {df_v["Efficiency"].max():.2f} at '
                                                             f'{max_eff_torque:.2f} N-m and {max_eff_rpm} RPM')
    #add stall torque to legend for each voltage
    ax.scatter(stall_torque, 0, label=f'Stall Torque: {stall_torque:.2f} N-m')
    
    plt.legend()
    ax.set_ylim(-0.05, df_v['Efficiency'].max() + 0.1)
    ax.set_xlabel('Torque (N-m)')
    ax.set_ylabel('Efficiency')
    plt.title(f'Efficiency vs Torque for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/Efficiency_vs_Torque/Efficiency_vs_Torque_{voltage}V.png'))
#plt.show()

# Plot Mechanical Power vs Efficiency for each voltage
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
    fig, ax = plt.subplots()
    coeffs_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    coeffs_Current_vs_RPM = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Current Draw (A)'], 1)
    stall_torque = coeffs_RPM_vs_Torque[1]
    
    print(f"Voltage: {voltage}")
    print(f"Stall Torque: {stall_torque}")
    print(f"RPM vs Torque Coefficients: {coeffs_RPM_vs_Torque}")
    print(f"Current vs RPM Coefficients: {coeffs_Current_vs_RPM}")

    torque_at_rpm = stall_torque + coeffs_RPM_vs_Torque[0] * df_v['Tach Reading (RPM)']
    current_at_rpm = coeffs_Current_vs_RPM[0] * df_v['Tach Reading (RPM)'] + coeffs_Current_vs_RPM[1]
    power_output = torque_at_rpm * df_v['Tach Reading (RPM)'] * (2 * np.pi / 60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input
    RPM_range = np.linspace(df_v['Tach Reading (RPM)'].min(), df_v['Tach Reading (RPM)'].max(), len(eff))
    
    ax.scatter(power_output, eff,
               label=r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} + B\omega)\omega}{(A\omega + \omega_{0})V}$")
    ax.scatter(df_v['Mechanical Power (W)'], df_v['Efficiency'],
               label=r'Direct from Data: $\eta = \frac{\tau\omega}{I V}$')
    plt.legend()
    ax.set_ylim(0, max(eff) + 0.1)
    ax.set_xlabel('Mechanical Power (W)')
    ax.set_ylabel('Efficiency')
    plt.title(f'Efficiency vs Mechanical Power for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/MechPwr_Vs_Efficiency/Efficiency_vs_Power_{voltage}V.png'))
#plt.show()

# Save motor properties to a text file and print its contents
with open('Motor_Properties.txt', 'w') as f:
    f.write("Properties of the motor:\n")
    f.write("The mean Motor Torque Constant is: " + str(np.mean(torque_constant)) + " N-m/A\n")
    f.write("The peak efficiency is " + str(DA.df['Efficiency'].max()) +
            " at a voltage of " + str(DA.df['Applied Voltage (v)'][DA.df['Efficiency'].idxmax()]) + "\n")
    f.write("The mean Back EMF constant is: " + str(np.mean(K_e)) + " V-s/rad\n")
    f.write("The average resistance is: " + str(DA.average_resistance) + " Ohms\n")

with open('Motor_Properties.txt', 'r') as f:
    print(f.read())

print(f"The maximum efficiency is {DA.df['Efficiency'].max()} at a voltage of "
      f"{DA.df['Applied Voltage (v)'][DA.df['Efficiency'].idxmax()]} V and occurs at a speed of "
      f"{DA.df['Tach Reading (RPM)'][DA.df['Efficiency'].idxmax()]} RPM and a torque of "
      f"{DA.df['Torque (N-m)'][DA.df['Efficiency'].idxmax()]} N-m")

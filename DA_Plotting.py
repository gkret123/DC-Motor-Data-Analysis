import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import GK_Data_Analysis as DA

my_path = os.path.dirname(os.path.abspath(__file__))
sns.set()

# Increase font sizes for all plots globally
plt.rcParams.update({
    'axes.labelsize': 16,    # X and Y labels
    'xtick.labelsize': 14,   # X tick labels
    'ytick.labelsize': 14,   # Y tick labels
    'axes.titlesize': 10     # Title size (optional)
})

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
fig, ax = plt.subplots()
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
fig, ax = plt.subplots()
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
fig, ax = plt.subplots()
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
fig, ax = plt.subplots()
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

    # Create a smooth RPM range and calculate smooth efficiency using trendline coefficients
    num_points = 100
    rpm_range = np.linspace(df_v['Tach Reading (RPM)'].min(), df_v['Tach Reading (RPM)'].max(), num_points)
    torque_smooth = stall_torque + coeffs_RPM_vs_Torque[0] * rpm_range
    current_smooth = coeffs_Current_vs_RPM[0] * rpm_range + coeffs_Current_vs_RPM[1]
    power_output_smooth = torque_smooth * rpm_range * (2 * np.pi / 60)
    power_input_smooth = current_smooth * voltage
    eff_smooth = power_output_smooth / power_input_smooth

    # Plot smooth trendline for efficiency
    ax.plot(rpm_range, eff_smooth,
            label=r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} + B\omega)\omega}{(A\omega + \omega_{0})V}$", color='red')
    # Plot direct data for efficiency
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Efficiency'],
               label=r'Direct from Data: $\eta = \frac{\tau\omega}{I V}$')

    # Add max efficiency point details in the legend
    max_eff_index = df_v['Efficiency'].idxmax()
    max_eff_torque = df_v['Torque (N-m)'][max_eff_index]
    max_eff_rpm = df_v['Tach Reading (RPM)'][max_eff_index]
    ax.scatter(max_eff_rpm, df_v['Efficiency'].max(),
               label=f'Max Efficiency: {df_v["Efficiency"].max():.2f} at {max_eff_torque:.2f} N-m and {max_eff_rpm} RPM')

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

    # Derive a smooth torque_range and corresponding rpm_range using the linear relationship
    num_points = 100
    torque_range = np.linspace(df_v['Torque (N-m)'].min(), df_v['Torque (N-m)'].max(), num_points)
    rpm_range = (torque_range - stall_torque) / coeffs_RPM_vs_Torque[0]
    current_at_rpm = coeffs_Current_vs_RPM[0] * rpm_range + coeffs_Current_vs_RPM[1]
    power_output = torque_range * rpm_range * (2 * np.pi / 60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input
    
    ax.plot(torque_range, eff,
               label=r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} + B\omega)\omega}{(A\omega + \omega_{0})V}$", color='red')
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
    
    # Derive a smooth torque_range and corresponding rpm_range to compute smooth mechanical power vs efficiency
    num_points = 100
    torque_range = np.linspace(df_v['Torque (N-m)'].min(), df_v['Torque (N-m)'].max(), num_points)
    rpm_range = (torque_range - stall_torque) / coeffs_RPM_vs_Torque[0]
    current_at_rpm = coeffs_Current_vs_RPM[0] * rpm_range + coeffs_Current_vs_RPM[1]
    power_output_smooth = torque_range * rpm_range * (2 * np.pi / 60)
    power_input_smooth = current_at_rpm * voltage
    eff_smooth = power_output_smooth / power_input_smooth

    # Plot the smooth trendline using the trendline coefficients
    ax.plot(power_output_smooth, eff_smooth,
            label=r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} + B\omega)\omega}{(A\omega + \omega_{0})V}$", color='red')
    
    # Plot the direct data
    ax.scatter(df_v['Mechanical Power (W)'], df_v['Efficiency'],
               label=r'Direct from Data: $\eta = \frac{\tau\omega}{I V}$')
    
    # Add max efficiency point from data to the legend
    max_eff_index = df_v['Efficiency'].idxmax()
    max_eff_power = df_v['Mechanical Power (W)'][max_eff_index]
    ax.scatter(max_eff_power, df_v['Efficiency'].max(),
               label=f'Max Efficiency: {df_v["Efficiency"].max():.2f} at {max_eff_power:.2f} W')
    
    # Add stall torque to the legend (mechanical power is zero at stall)
    ax.scatter(0, 0, label=f'Stall Torque: {stall_torque:.2f} N-m')

    plt.legend()
    ax.set_ylim(0, max(eff_smooth) + 0.1)
    ax.set_xlabel('Mechanical Power (W)')
    ax.set_ylabel('Efficiency')
    plt.title(f'Efficiency vs Mechanical Power for {voltage} V')
    plt.savefig(os.path.join(my_path, f'Plots/MechPwr_Vs_Efficiency/Efficiency_vs_Power_{voltage}V.png'))
#plt.show()





# Plot the Z-scores in histgram/s
fig, ax = plt.subplots()
ax.hist(DA.df['Z_Score_Scale Reading (g)'], bins=20, alpha=0.5, label='Scale Reading (g)')
ax.hist(DA.df['Z_Score_Current Draw (A)'], bins=20, alpha=0.5, label='Current Draw (A)')
ax.hist(DA.df['Z_Score_Tach Reading (RPM)'], bins=20, alpha=0.5, label='Tach Reading (RPM)')

ax.set_xlabel('Z-Score')
ax.set_ylabel('Frequency')
ax.legend()
plt.title('Z-Scores for Scale Reading, Current Draw, and Tach Reading')
plt.savefig(os.path.join(my_path, 'Plots/Z_Scores.png'))


# Plot the outliers grouped by voltage for clarity
import matplotlib.cm as cm
# Get unique voltage levels from the dataset
voltages = sorted(DA.df['Applied Voltage (v)'].unique())
colors = cm.rainbow(np.linspace(0, 1, len(voltages)))

fig, ax = plt.subplots()
# Optionally, plot the full data in light gray for context
ax.scatter(DA.df['Tach Reading (RPM)'], DA.df['Torque (N-m)'], color='lightgray', label='Data', alpha=0.5)

# Plot outliers for each voltage level with a unique color
for i, voltage in enumerate(voltages):
    # Filter outliers for the current voltage (assumes the 'Applied Voltage (v)' column exists in outliers_Torque)
    outliers_voltage = DA.outliers_Torque[DA.outliers_Torque['Applied Voltage (v)'] == voltage]
    ax.scatter(outliers_voltage['Tach Reading (RPM)'], outliers_voltage['Torque (N-m)'],
               color=colors[i], edgecolor='black', s=100,
               label=f'Outlier(s) at {voltage} V')

ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N-m)')
plt.title('Outliers in Torque vs RPM by Voltage')
plt.legend()
plt.savefig(os.path.join(my_path, 'Plots/Outliers_Torque_vs_RPM_by_Voltage.png'))


#list all outliers that were excluded
print("The following outliers were excluded from the data for having a Z-score > 2 (meaning the value was greater than 2 standard deviations above the mean):")
print("Outliers in the scale reading data: \n", DA.outliers_Scale)
print("Outliers in the Current Draw data: \n", DA.outliers_Current)
print("Outliers in the Tach Reading data: \n", DA.outliers_Tach)
print("\n")
print("These outliers were excluded due to them being too far from the line fit to the torque data. This is called a residual error: \n", DA.outliers_Torque)

# export outliers to text file
with open('Outliers.txt', 'w') as f:
    f.write("The following outliers were excluded from the data for having a Z-score > 2 (meaning the value was greater than 2 standard deviations above the mean):\n")
    f.write("Outliers in the scale reading data: \n" + str(DA.outliers_Scale) + "\n")
    f.write("Outliers in the Current Draw data: \n" + str(DA.outliers_Current) + "\n")
    f.write("Outliers in the Tach Reading data: \n" + str(DA.outliers_Tach) + "\n")
    f.write("These outliers were excluded due to them being too far from the line fit to the torque data. This is called a residual error: \n" + str(DA.outliers_Torque) + "\n")


        
    #print stall torque data at each voltage
    for voltage in DA.df['Applied Voltage (v)'].unique():
        df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
        coeffs_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
        stall_torque = coeffs_RPM_vs_Torque[1]
        print(f"For {voltage} V, the stall torque is {stall_torque:.3f} N-m")

    #print the no load speed for each voltage
    for voltage in DA.df['Applied Voltage (v)'].unique():
        df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Current Draw (A)')
        coeffs_Current_vs_RPM = np.polyfit(df_v['Current Draw (A)'], df_v['Tach Reading (RPM)'], 1)
        no_load_speed = coeffs_Current_vs_RPM[1]
        print(f"For {voltage} V, the no load speed is {no_load_speed:.3f} RPM")
        
    # New code: Tabulate all the motor performance data in a single table

    table_data = []
    for voltage in sorted(DA.df['Applied Voltage (v)'].unique()):
        df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]

        # Maximum Efficiency Data
        max_eff = df_v['Efficiency'].max()
        idx = df_v['Efficiency'].idxmax()
        max_eff_rpm = df_v.loc[idx, 'Tach Reading (RPM)']
        max_eff_torque = df_v.loc[idx, 'Torque (N-m)']

        # Stall Torque Data
        df_v_sorted = df_v.sort_values(by='Tach Reading (RPM)')
        coeffs_RPM_vs_Torque = np.polyfit(df_v_sorted['Tach Reading (RPM)'], df_v_sorted['Torque (N-m)'], 1)
        stall_torque = coeffs_RPM_vs_Torque[1]

        # No Load Speed Data
        df_v_current = df_v.sort_values(by='Current Draw (A)')
        coeffs_Current_vs_RPM = np.polyfit(df_v_current['Current Draw (A)'], df_v_current['Tach Reading (RPM)'], 1)
        no_load_speed = coeffs_Current_vs_RPM[1]

        table_data.append({
            'Voltage (V)': voltage,
            'Max Efficiency': round(max_eff, 3),
            'Max Eff. RPM': round(max_eff_rpm, 1),
            'Max Eff. Torque (N-m)': round(max_eff_torque, 3),
            'Stall Torque (N-m)': round(stall_torque, 3),
            'No Load RPM': round(no_load_speed, 3)
        })
with open('Motor_Properties.txt', 'w') as f:
    f.write("Properties of the motor:\n")
    f.write("The mean Motor Torque Constant is: " + str(np.mean(torque_constant)) + " N-m/A\n")
    f.write("The peak efficiency is " + str(DA.df['Efficiency'].max()) +
            " at a voltage of " + str(DA.df['Applied Voltage (v)'][DA.df['Efficiency'].idxmax()]) + "\n")
    f.write("The mean Back EMF constant is: " + str(np.mean(K_e)) + " V-s/rad\n")
    f.write("The average resistance is: " + str(DA.average_resistance) + " Ohms\n")

# Save motor properties to a text file and print its contents
with open('Motor_Properties.txt', 'w') as f:
    f.write("Properties of the motor:\n")
    f.write("The mean Motor Torque Constant is: " + str(np.mean(torque_constant)) + " N-m/A\n")
    f.write("The peak efficiency is " + str(DA.df['Efficiency'].max()) +
            " at a voltage of " + str(DA.df['Applied Voltage (v)'][DA.df['Efficiency'].idxmax()]) + "\n")
    f.write("The mean Back EMF constant is: " + str(np.mean(K_e)) + " V-s/rad\n")
    f.write("The average resistance is: " + str(DA.average_resistance) + " Ohms\n")

    f.write("\nTabulated Motor Performance Data:\n")
    f.write(pd.DataFrame(table_data).to_string(index=False))

with open('Motor_Properties.txt', 'r') as f:
    print(f.read())
    # Print the maximum efficiency for each voltage level
    for voltage in DA.df['Applied Voltage (v)'].unique():
        df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
        print(f"For {voltage} V, the maximum efficiency is {df_v['Efficiency'].max():.3f} at a speed of "
              f"{df_v['Tach Reading (RPM)'][df_v['Efficiency'].idxmax()]:.1f} RPM and a torque of "
              f"{df_v['Torque (N-m)'][df_v['Efficiency'].idxmax()]:.3f} N-m")
        
    df_table = pd.DataFrame(table_data)
    print("\nTabulated Motor Performance Data:")
    print(df_table.to_string(index=False))

#plot the no load speed for each voltage using the tabulated motor performance data
fig, ax = plt.subplots()
df_table = pd.DataFrame(table_data)
ax.scatter(df_table['Voltage (V)'], df_table['No Load RPM'], color='blue')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('No Load Speed (RPM)')
plt.title('No Load Speed vs Voltage')
plt.savefig(os.path.join(my_path, 'Plots/No_Load_Speed_vs_Voltage.png'))

#plot the stall torque for each voltage using the tabulated motor performance data
fig, ax = plt.subplots()
ax.scatter(df_table['Voltage (V)'], df_table['Stall Torque (N-m)'], color='red')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Stall Torque (N-m)')
plt.title('Stall Torque vs Voltage')
plt.savefig(os.path.join(my_path, 'Plots/Stall_Torque_vs_Voltage.png'))

#plot the peak efficiency for each voltage using the tabulated motor performance data
fig, ax = plt.subplots()
ax.scatter(df_table['Voltage (V)'], df_table['Max Efficiency'], color='green')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Peak Efficiency')
plt.title('Peak Efficiency vs Voltage')
plt.savefig(os.path.join(my_path, 'Plots/Peak_Efficiency_vs_Voltage.png'))


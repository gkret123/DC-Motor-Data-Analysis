import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import GK_Data_Analysis as DA
"""
#plot speed vs efficiency for each applied voltage
for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage]
    fig, ax = plt.subplots()
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Efficiency'])
    ax.set_xlabel('RPM')
    ax.set_ylabel('Efficiency')
    plt.title(f'Efficiency vs RPM for {voltage} V')
    #plt.savefig(os.path.join(my_path, f'Plots/Efficiency_vs_RPM/Efficiency_vs_RPM_{voltage}V.png'))
plt.show()
"""
# eff = (stall_torque - coeff_current_vs_torque[1])*speed/(coeff_current_vs_RPM[0]*voltage)
#plot efficiency vs speed for each applied voltage using the above formula

for voltage in DA.df['Applied Voltage (v)'].unique():
    df_v = DA.df[DA.df['Applied Voltage (v)'] == voltage].sort_values(by='Tach Reading (RPM)')
    fig, ax = plt.subplots()
    coefficients_RPM_vs_Torque = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Torque (N-m)'], 1)
    coefficients_Current_vs_RPM = np.polyfit(df_v['Tach Reading (RPM)'], df_v['Current Draw (A)'], 1)
    stall_torque = coefficients_RPM_vs_Torque[1]
    print(f"Voltage: {voltage}")
    print(f"Stall Torque: {stall_torque}")
    print(f"RPM vs Torque Coefficients: {coefficients_RPM_vs_Torque}")
    print(f"Current vs RPM Coefficients: {coefficients_Current_vs_RPM}")

    torque_at_rpm = stall_torque + coefficients_RPM_vs_Torque[0] * df_v['Tach Reading (RPM)']
    current_at_rpm = coefficients_Current_vs_RPM[0] * df_v['Tach Reading (RPM)'] + coefficients_Current_vs_RPM[1]
    power_output = torque_at_rpm * df_v['Tach Reading (RPM)'] * (2*np.pi/60)
    power_input = current_at_rpm * voltage
    eff = power_output / power_input
    RPM_range = np.linspace(df_v['Tach Reading (RPM)'].min(), df_v['Tach Reading (RPM)'].max(), len(eff))
    ax.scatter(RPM_range, eff, label = r"From Trendline Coeff: $\eta = \frac{(\tau_{stall} \cdot + B\omega)\cdot \omega}{(A\omega + \omega_{0})*V}$ ")
    #plot direct efficiency calculation
    ax.scatter(df_v['Tach Reading (RPM)'], df_v['Efficiency'], label = r'Direct from Data: $\eta = \frac{\tau \cdot \omega}{I \cdot V}$')
    plt.legend()
    ax.set_ylim(0, max(eff)+0.1)
    ax.set_xlabel('RPM')
    ax.set_ylabel('Efficiency')
    plt.title(f'Efficiency vs RPM for {voltage} V')
plt.show()
    #plt.savefig(os.path.join(my_path, f'Plots/Efficiency_vs_RPM/Efficiency_vs_RPM_{voltage}V.png')
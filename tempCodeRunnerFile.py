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

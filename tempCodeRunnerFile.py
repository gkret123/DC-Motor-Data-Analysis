#plot outliers for single dataset and create lines that are 2 std offset on either side of the best fit line for the 5V dataset
fig, ax = plt.subplots(figsize = size_of_plots)
df_5v = DA.df_original[DA.df_original['Applied Voltage (v)'] == 5].sort_values(by='Tach Reading (RPM)')
ax.scatter(df_5v['Tach Reading (RPM)'], df_5v['Torque (N-m)'], color='blue', label='Data')
ax.errorbar(df_5v['Tach Reading (RPM)'], df_5v['Torque (N-m)'],
            yerr=abs(df_5v['Relative Error Torque']), fmt='o', color='blue')
coeffs_RPM_vs_Torque = np.polyfit(df_5v['Tach Reading (RPM)'], df_5v['Torque (N-m)'], 1)
poly = np.poly1d(coeffs_RPM_vs_Torque)
ax.plot(df_5v['Tach Reading (RPM)'], poly(df_5v['Tach Reading (RPM)']), color='red', label='Best Fit Line')
ax.fill_between(df_5v['Tach Reading (RPM)'].values, poly(df_5v['Tach Reading (RPM)'].values) - 1.5 * DA.std_resid,
                poly(df_5v['Tach Reading (RPM)'].values) + 1.5 * DA.std_resid, color='gray', alpha=0.5)
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N-m)')
plt.title('Outliers in Torque vs RPM for 5V')
plt.legend()
plt.savefig(os.path.join(my_path, 'Plots/Outliers_Torque_vs_RPM_5V.png'))
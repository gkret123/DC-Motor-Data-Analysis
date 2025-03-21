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
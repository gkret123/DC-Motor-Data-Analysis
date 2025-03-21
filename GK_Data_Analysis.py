import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        lambda row: 0.00981 * (row['Scale Reading (g)'] - row['Lever arm load  (end) (g)'])* 0.01 * row['Moment arm (cm)'], axis=1 # took out "- row['Lever arm load  (end) (g)']"
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

#
#drop all data associated with the 10.5 volt data point
df = df[df['Applied Voltage (v)'] != 10.5]

#find outliers for all relevant columns and returnt the coutliers, the z score and the column name
Z_score_lim = 2.5
def find_outliers(df, column):
    Z_Score = (df[column] - df[column].mean()) / df[column].std()
    #add the z score to the dataframe
    df[f'Z_Score_{column}'] = Z_Score
    outliers = df[np.abs(Z_Score) > Z_score_lim]
    return outliers, Z_Score

#execute the function 
outliers_Scale, Z_Score_Scale = find_outliers(df, 'Scale Reading (g)')
print("Outliers in the scale reading data: \n", outliers_Scale)
outliers_Current, Z_Score_Current = find_outliers(df, 'Current Draw (A)')
print("Outliers in the Current Draw data: \n", outliers_Current)
outliers_Tach, Z_Score_Tach = find_outliers(df, 'Tach Reading (RPM)')
print("Outliers in the Tach Reading data: \n", outliers_Tach)
outliers_Torque, Z_Score_Torque = find_outliers(df, 'Torque (N-m)')
print("Outliers in the Torque data: \n", outliers_Torque)
outliers_Efficiency, Z_Score_Efficiency = find_outliers(df, 'Efficiency')
print("Outliers in the Efficiency data: \n", outliers_Efficiency)

#use the z-score df columns to remove all rows with outliers from the data for each row and z score
if 'Z_Score_Scale Reading (g)' in df.columns:
    df = df[np.abs(df['Z_Score_Scale Reading (g)']) < Z_score_lim]
    print("Removed outliers in Scale Reading (g) \n")
if 'Z_Score_Current Draw (A)' in df.columns:
    df = df[np.abs(df['Z_Score_Current Draw (A)']) < Z_score_lim]
    print("Removed outliers in Current Draw (A) \n")
if 'Z_Score_Tach Reading (RPM)' in df.columns:
    df = df[np.abs(df['Z_Score_Tach Reading (RPM)']) < Z_score_lim]
    print("Removed outliers in Tach Reading (RPM) \n")
    
    #Z-score isnt enough, the data is not normally distributed, so we will use the standardized residuals
    # Apply the outlier removal strategy for Torque on a per-voltage basis
    from scipy.stats import linregress

    # Create a global outlier mask for Torque outliers across all voltage groups
    global_outlier_mask = pd.Series(False, index=df.index)
    for voltage in df['Applied Voltage (v)'].unique():
        # Subset the dataframe for the current voltage
        sub = df[df['Applied Voltage (v)'] == voltage]
        # Fit a regression line for Torque vs Tach Reading within this voltage
        slope, intercept, r_value, p_value, std_err = linregress(sub['Tach Reading (RPM)'], sub['Torque (N-m)'])
        # Calculate predicted torque and residuals
        predicted_torque = slope * sub['Tach Reading (RPM)'] + intercept
        residuals = sub['Torque (N-m)'] - predicted_torque
        sigma_tau = residuals.std()
        # Calculate standardized residuals
        std_resid = residuals / sigma_tau
        # Save the computed values back into the dataframe
        df.loc[sub.index, 'Predicted_Torque'] = predicted_torque
        df.loc[sub.index, 'Residuals'] = residuals
        df.loc[sub.index, 'Standardized Residuals'] = std_resid
        # Identify outliers using the threshold (absolute standardized residual >= 1.5)
        group_outlier_mask = np.abs(std_resid) >= 1.5
        global_outlier_mask[sub.index] = group_outlier_mask
    # Extract and display the outliers for the Torque data across all voltage groups
    outliers_Torque = df[global_outlier_mask]
    print("Outliers in the Torque data for each voltage:")
    print(outliers_Torque)
    # Remove the outliers from the dataframe
    df = df[~global_outlier_mask]
print("\nRemoved outliers in Torque (N-m).")

#save to new excel file
df.to_excel("GK_Data_M1_filled.xlsx", index=False)

print("Data Processing Complete")


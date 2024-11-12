import pandas as pd

# Load the crop yield and temperature data CSV files
df_crop = pd.read_csv('Datasets/crop_yield.csv')   # Replace 'crop_yield.csv' with the actual path to your crop yield file
df_temp = pd.read_csv('Datasets/Temp_dataset_1.csv')    # Replace 'temp_data.csv' with the actual path to your temperature file

df_temp.columns = df_temp.columns.str.strip()
# Perform the merge based on State and Crop_Year columns
merged_df = pd.merge(df_crop, df_temp, on=['Crop_Year','State'], how='left')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)  # This will save it as 'merged_data.csv'

# Display the first few rows of the merged dataframe to confirm the merge
print(merged_df.head(100))

import pandas as pd

df = pd.read_csv('C:/Data/trials.csv') # Replace with file path
df['Date'] = pd.to_datetime(df['Date']) # Convert date/time format
df['Date'] = df['Date'].dt.date # Extract only the date

# Group_by_Species_&_Date,_calculate_mean--------------------------- #
daily_avg = df.groupby(['Species', 'Date'], as_index=False).agg({
    'Stomatal_C': 'mean',
    'Moisture': 'mean',
    'Temperature': 'mean'
})

# Save_to_new_CSV_file---------------------------------------------- #
daily_avg.to_csv('daily_averages.csv', index=False)
# Display_preview--------------------------------------------------- #
print("Daily averages calculated successfully! Preview:")
print(daily_avg.head())
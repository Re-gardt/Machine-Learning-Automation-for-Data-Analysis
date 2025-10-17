import os
import pandas as pd

# --- Folder containing your CSV files -------------------------------------------------- #
data_folder = 'C:/Data'
output_folder = os.path.join(data_folder, 'daily_averages')
os.makedirs(output_folder, exist_ok=True)

# --- Loop through all CSV files in the folder ----------------------------------------- #
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path)

        # --- Convert 'Date' column to datetime and keep only date ---------------------- #
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y').dt.date

        # --- Automatically detect numeric columns for averaging ---------------------- #
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # --- Group by 'Species' and 'Date', calculate mean for all numeric columns --- #
        daily_avg = df.groupby(['Species', 'Date'], as_index=False)[numeric_cols].mean()

        # --- Save output CSV in the output folder ------------------------------------ #
        output_file = os.path.join(data_folder, f'daily_avg_{file_name}')
        daily_avg.to_csv(output_file, index=False)

        print(f"Processed {file_name} → saved daily averages to {output_file}")

print("All CSV files processed successfully!")

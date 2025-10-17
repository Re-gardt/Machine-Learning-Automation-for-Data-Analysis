import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ----------------------------------------------------------------------- #
DATA_FOLDER = 'C:/Data'
OUTPUT_DIR = os.path.join(DATA_FOLDER, 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Seaborn style ------------------------------------------------------------------------ #
sns.set_theme(style="whitegrid")

# --- Loop through all CSV files in the folder -------------------------------------------- #
csv_files = glob.glob(os.path.join(DATA_FOLDER, '*.csv'))

for file_path in csv_files:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    df = df.dropna(subset=['Date'])  # drop rows with invalid dates

    # --- Detect columns ----------------------------------------------------------------- #
    hue_col = 'Species' if 'Species' in df.columns else None
    plot_cols = df.select_dtypes(include='number').columns.drop([hue_col] if hue_col else [], errors='ignore')

    # --- Plot each numeric column -------------------------------------------------------- #
    for col in plot_cols:
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=df, x='Date', y=col, hue=hue_col, marker='o', legend='auto')
        plt.title(f'Daily {col} per {hue_col or "dataset"}')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Save plot
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

print(f" Saved plots for {len(csv_files)} CSV files to: {OUTPUT_DIR}")
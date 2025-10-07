import pandas as pd
import glob
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
# CSV_Folder_Selection-------------------------------------------------------- #
folder_path = 'C:/Data/'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
dfs = {}
for i, file in enumerate(csv_files):
    try:
        dfs[f'data_frame_{i+1}'] = pd.read_csv(file)
        print(f"Loaded: {file}")
    except Exception as e:
        print(f"Error reading {file}: {e}")
# DEFINE_MODELS--------------------------------------------------------------- #
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}
# Hyperparameter-------------------------------------------------------------- #
parameter = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 7],
        'learning_rate': [0.01, 0.1]
    }
}
# EVALUATION------------------------------------------------------------------ #
def evaluate_model(model, df_n, test_s, y_true, y_predict):
    print(f'\n{model} Results on {df_n} (Test size = {test_s}):')
    print(f'MAE: {mean_absolute_error(y_true, y_predict):.2f}')
    print(f'MSE: {mean_squared_error(y_true, y_predict):.2f}')
    print(f'R2: {r2_score(y_true, y_predict):.2f}')
# Result_collection ---------------------------------------------------------- #
results = []
search_type = 'grid'
# Dataframe_Loop-------------------------------------------------------------- #
for df_name, df in dfs.items():
    print(f"\n Checking dataset: {df_name} | Shape: {df.shape}") # Check_1
    if df.shape[1] < 5:
        print(f"{df_name} has fewer than 5 columns. Skipping.")
        continue
    X = df.iloc[:, 4:]
    y = df.iloc[:, 3]
    test_sizes = [0.3,0.4,0.5]
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)
        # Model_Training_&_Evaluation------------------------------------------- #
        for model_name, base_model in models.items():
            param_grid = parameter[model_name]
            if search_type == 'grid':
                search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, scoring='r2')
            else:
                search = RandomizedSearchCV(base_model, param_grid, n_iter=10, cv=3, n_jobs=-1, scoring='r2', random_state=42)
            try:
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                y_prediction = best_model.predict(X_test)
            # Evaluation_and_logging_for_save&print-------------------------------- #
                mae = mean_absolute_error(y_test, y_prediction)
                mse = mean_squared_error(y_test, y_prediction)
                r2 = r2_score(y_test, y_prediction)
            # Best_Parameter_Performance------------------------------------------- #
                print(f'\nBest Parameters for {model_name} on {df_name} (Test size = {test_size}): {search.best_params_}')
                evaluate_model(model_name, df_name, test_size, y_test, y_prediction)
            # Define_result-------------------------------------------------------- #
                results.append({
                    'Dataset': df_name,
                    'Model': model_name,
                    'Test Size': test_size,
                    'Best Parameters': search.best_params_,
                    'MAE': round(mae, 4),
                    'MSE': round(mse, 4),
                    'R2': round(r2, 4)
                })
            except Exception as e:
                print(f"Error training {model_name} on {df_name} (Test size = {test_size}): {e}")
# Save_Results_to_CSV---------------------------------------------------------- #
print(f"\n Total results collected: {len(results)}") # Check_2
if results:
    print("Sample result:", results[0])  # Check_3
if results:
    output_path = os.path.join(folder_path, 'model_results.csv')
    try:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\n Saved results to: {output_path}")
    except Exception as e:
        print(f" Failed to save CSV: {e}")
else:
    print("\n No results were generated. Please check your data.")
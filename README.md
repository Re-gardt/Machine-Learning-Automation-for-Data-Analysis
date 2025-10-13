# Machine-Learning-Automation-for-Data-Analysis
I am new to this and learning Python

The script assumes the data has been cleaned beforehand and is hard-coded to target columns "y = df.iloc[:, 3] and "X = df.iloc[:, 4:]", assuming the 4th column is the target variable which may be an issue. I intend to fix this at some point.
The single train/test split may lead to high variation
Hyper parameters may be limited

Changes for future reference
- Add preprocessing steps
- Use cross-validation scoring on the final tuned model or average results over multiple splits

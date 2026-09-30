# Heart Failure 30-Day Readmission Prediction

## 🚀 Live Streamlit App

[Click here to open the live application](https://heart-failure-readmission5.streamlit.app/)# Heart Failure 30-Day Readmission Prediction

## Files
- `Heart_Failure_30_Day_Readmission_Project.ipynb` — complete reproducible notebook
- `model_comparison.csv` — final test metrics and confusion-matrix counts
- `project_summary.txt` — concise results
- `01_target_distribution.png` through `07_knn_hyperparameter.png` — required visuals
- `dataset_12000_records.csv` — input dataset (keep with notebook if submitting the project folder)

## How to run
1. Put the CSV in the same folder as the notebook.
2. Open the notebook in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
3. Run all cells from top to bottom.
4. The notebook performs preprocessing, cross-validation, final evaluation, plots, and written analysis.

## Main result on the supplied dataset
Logistic Regression produced the strongest overall test performance:
- Accuracy: ~0.895
- Precision: ~0.852
- Recall: ~0.786
- F1: ~0.818
- ROC-AUC: ~0.956

These numbers are specific to the supplied dataset and split (80/20, stratified, random_state=42).

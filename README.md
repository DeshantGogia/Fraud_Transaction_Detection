# Fraud Transaction Detection

This project builds a supervised fraud detection pipeline for financial transactions. It includes feature engineering, SMOTE resampling for class imbalance, model training and evaluation with XGBoost/LightGBM, and a saved XGBoost model for inference.

Files in this repository

- `preprocessing_model_building.ipynb` - Notebook with preprocessing, feature engineering, model training, evaluation, and saving the model.
- `app.py` - Example app (if present) to load the saved model and serve predictions.
- `dataset/` - Raw dataset (ignored by default via `.gitignore`).
- `model/` - Saved model artifacts (ignored by default via `.gitignore`).
- 
<img width="1058" height="851" alt="Screenshot 2025-09-27 232829" src="https://github.com/user-attachments/assets/e7f69b71-8e37-48f5-8f12-42b60f79913a" />

<img width="1102" height="850" alt="Screenshot 2025-09-27 232628" src="https://github.com/user-attachments/assets/17c3d7f2-3086-4948-99a1-f745a03e161c" />

<img width="1358" height="895" alt="Screenshot 2025-09-27 232704" src="https://github.com/user-attachments/assets/b9c085ec-e7dd-443d-ac4f-c72d5af47f3b" />


Requirements

See `requirements.txt` for required Python packages.

Quick start — push to GitHub

1. Create a new repository on GitHub (via the website or `gh` CLI).

2. On your local machine, add the remote and push:

   ```powershell
   # Replace USERNAME and REPO with your GitHub username and repo name
   git remote add origin https://github.com/USERNAME/REPO.git
   git branch -M main
   git push -u origin main
   ```

   Or, if you have the GitHub CLI installed:

   ```powershell
   gh repo create REPO --public --source=. --remote=origin --push
   ```

Notes

- The notebook currently uses an absolute path to the CSV. Update it to a relative path (`dataset/Fraud.csv`) so others can run the notebook after cloning.
- If `git commit` fails because user.name/email are not set, configure them with:

   ```powershell
   git config user.name "Your Name"
   git config user.email "you@example.com"
   ```

Short project description

This repository contains a production-ready fraud detection pipeline that trains and evaluates gradient-boosted models (XGBoost and LightGBM) on engineered transaction features and provides a saved XGBoost model for inference.

# Fraud Transaction Detection

This project builds a supervised fraud detection pipeline for financial transactions. It includes feature engineering, SMOTE resampling for class imbalance, model training and evaluation with XGBoost/LightGBM, and a saved XGBoost model for inference.

Files in this repository

- `preprocessing_model_building.ipynb` - Notebook with preprocessing, feature engineering, model training, evaluation, and saving the model.
- `app.py` - Example app (if present) to load the saved model and serve predictions.
- `dataset/` - Raw dataset (ignored by default via `.gitignore`).
- `model/` - Saved model artifacts (ignored by default via `.gitignore`).

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

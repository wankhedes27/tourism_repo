# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# -----------------------------
# Authentication and Constants
# -----------------------------
# --- 🔐 Load your HF token safely ---
token = os.getenv("HF_TOKEN")
if not token:
    sys.exit("❌ HF_TOKEN not found. Please set it in your Colab before running this script.")

# --- Initialize API client ---
api = HfApi(token=token)


# Dataset location on Hugging Face
DATASET_PATH = "hf://datasets/wankhedes27/tourism-repo/tourism.csv"

# -----------------------------
# Load Dataset
# -----------------------------
tourism_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {tourism_df.shape}")

# -----------------------------
# Define Target Variable
# -----------------------------
target = 'ProdTaken'  # Whether the customer purchased the product (1 = Yes, 0 = No)

# -----------------------------
# Define Feature Columns
# -----------------------------
# Numeric features
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# Categorical features
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# -----------------------------
# Data Cleaning
# -----------------------------
# Strip spaces and make text consistent (lowercase or title case)
for col in categorical_features:
    tourism_df[col] = tourism_df[col].astype(str).str.strip().str.title()

# Specific correction for known inconsistencies
tourism_df['Gender'] = tourism_df['Gender'].replace({
    'Fe Male': 'Female',
    'Female': 'Female',
    'Male': 'Male'
})

# -----------------------------
# Define Predictors and Target
# -----------------------------
X = tourism_df[numeric_features + categorical_features]
y = tourism_df[target]

# -----------------------------
# Split Data into Train/Test
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Save Split Data to CSV
# -----------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train/test split completed and files saved locally.")

# -----------------------------
# Upload Files to Hugging Face
# -----------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="wankhedes27/tourism-repo",
        repo_type="dataset",
    )

print("All files uploaded successfully to Hugging Face dataset repository.")

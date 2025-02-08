import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv("C:/Users/jenit/Downloads/bit(2)/sample_resume_evaluation.csv")  # Update with your actual file path

# Check if the necessary columns are present
if 'JobDescription' not in data.columns or 'ResumeDescription' not in data.columns:
    raise ValueError("The dataset must contain 'JobDescription' and 'ResumeDescription' columns.")

if 'SimilarityScore' not in data.columns:
    raise ValueError("The dataset must contain a 'SimilarityScore' column with ground truth values.")

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert job descriptions and resume descriptions to embeddings
job_embeddings = model.encode(data['JobDescription'].tolist(), convert_to_tensor=True)
resume_embeddings = model.encode(data['ResumeDescription'].tolist(), convert_to_tensor=True)

# Compute cosine similarity
predicted_scores = util.cos_sim(job_embeddings, resume_embeddings).diagonal().tolist()

# Ground truth similarity scores
ground_truth_scores = data['SimilarityScore'].tolist()

# Ensure ground truth scores are normalized between 0 and 1
if not all(0 <= score <= 1 for score in ground_truth_scores):
    raise ValueError("The 'SimilarityScore' column must have values between 0 and 1.")

# 1. Mean Squared Error (MSE)
mse = mean_squared_error(ground_truth_scores, predicted_scores)
print(f"Mean Squared Error (MSE): {mse}")

# 2. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

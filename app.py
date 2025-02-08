from flask import Flask, request, render_template
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2

app = Flask(__name__)

# Directory to store uploaded resumes
RESUMES_DIR = "resumes"
os.makedirs(RESUMES_DIR, exist_ok=True)  # Create the resumes directory if it doesn't exist

# Initialize the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function to extract candidate name from file name
def extract_name_from_filename(file_name):
    match = re.search(r" - (.+?)\.pdf", file_name)
    if match:
        return match.group(1)
    return file_name.replace(".pdf", "")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vacant_positions', methods=['POST'])
def vacant_positions():
    # Get the number of vacant positions
    num_positions = int(request.form['num_positions'])
    return render_template('job_descriptions.html', num_positions=num_positions)

@app.route('/upload', methods=['POST'])
def upload_files():
    print(f"Received num_positions: {request.form.get('num_positions')}")
    
    # Get job descriptions from the form
    job_descriptions = [request.form[f'job_desc_{i}'] for i in range(int(request.form['num_positions']))]
    print(f"Job Descriptions: {job_descriptions}")

    # Upload resumes
    resume_files = request.files.getlist('resumes')  # Get resumes
    resumes = []
    candidate_names = []
    for resume_file in resume_files:
        if resume_file and resume_file.filename.endswith('.pdf'):
            file_path = os.path.join(RESUMES_DIR, resume_file.filename)
            resume_file.save(file_path)
            resume_text = extract_text_from_pdf(file_path)
            if resume_text.strip():
                resumes.append(resume_text)
                candidate_names.append(extract_name_from_filename(resume_file.filename))

    # Check if job descriptions and resumes are provided
    if not job_descriptions or not resumes:
        return "Please provide job descriptions and resumes.", 400

    # Convert job descriptions and resumes to embeddings
    job_embeddings = model.encode(job_descriptions)
    resume_embeddings = model.encode(resumes)

    # Match each job description with resumes
    results = []
    for i, job_desc in enumerate(job_descriptions):
        similarity_scores = cosine_similarity([job_embeddings[i]], resume_embeddings)[0]
        ranked_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
        matched_resumes = [
            {
                "rank": j + 1,
                "candidate_name": candidate_names[ranked_indices[j]],
                "score": similarity_scores[ranked_indices[j]]
            }
            for j in range(len(ranked_indices))
        ]
        results.append({
            "job_description": job_desc,
            "matched_resumes": matched_resumes[:3]  # Limit to top 3 matches
        })

    # Render results in the frontend
    return render_template('results_multiple.html', results=results)

# Test Route to check if form is working
@app.route('/test_upload', methods=['POST'])
def test_upload():
    print("Test upload triggered")
    return "Upload Successful"

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB
    app.run(debug=True)

import pandas as pd

# Define job descriptions and resume descriptions
data = [
    ("Software Engineer", "Develop and maintain software applications.", "Alice", "Experienced software developer skilled in Python."),
    ("Data Scientist", "Analyze and interpret complex data to help decision making.", "Bob", "Expertise in data analysis, machine learning, and AI."),
    ("Project Manager", "Lead project teams to ensure on-time delivery.", "Charlie", "Strong leadership skills with experience in project management."),
    ("UX Designer", "Design user interfaces for web applications.", "Dave", "UX/UI designer with experience in Figma and Adobe XD."),
    ("Marketing Specialist", "Create and execute marketing strategies.", "Eve", "Marketing professional with expertise in digital marketing.")
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["JobTitle", "JobDescription", "CandidateName", "ResumeDescription"])

# Save to CSV
csv_file_path = "C:/Users/jenit/Downloads/bit(2)/sample_resume_evaluation.csv"
df.to_csv(csv_file_path, index=False)

csv_file_path

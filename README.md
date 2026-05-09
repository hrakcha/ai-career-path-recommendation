AI Career Path Recommendation System:

A supervised machine learning project that recommends suitable IT career paths based on a user's skills using TF-IDF Vectorization, Cosine Similarity, and Logistic Regression.

About the Project:

Choosing the right career is one of the biggest challenges for students and freshers.
This system takes a user's skills as input and automatically recommends the most suitable IT career roles by matching them against a structured job skills dataset.

Dataset:

Type: Structured IT job skills dataset 

Records: 15 job roles

Domains: 9(AI/Data, Web, Cloud, Security, Mobile, Software, Design, Blockchain, Management)

Salary Range: 5.0 – 15.0 LPA

Job Roles Covered: Data Scientist, Machine Learning Engineer, Data Analyst, Web Developer, Backend Developer, Cloud Engineer, Cybersecurity Analyst, DevOps Engineer, Mobile App Developer, UI/UX Designer, Software Engineer, Database Administrator, QA Engineer, Blockchain Developer, Product Manager

Tech Stack:

1.Python

2.Pandas

3.NumPy

4.Scikit-learn (TF-IDF, Logistic Regression, LabelEncoder, Cosine Similarity)

5.Re (Regex for text cleaning)

How It Works:

1.Data Preprocessing:

Lowercasing text,
Removing stopwords,
Cleaning special characters,

2.Feature Extraction

TF-IDF Vectorization converts skill text into numerical vectors

3.Model Training

Logistic Regression classifier trained on labeled career data,
Each career role treated as a predefined output class

4.Career Recommendation

Calculates Cosine Similarity between user skill vector and career skill vectors,
Recommends top 3 matching career paths with percentage match score

5.Skill Gap Analysis

Identifies missing skills for the recommended career,
Suggests improvement areas for the user

Results:

Accuracy: 95%

Precision: 0.95

Recall; 0.95

F1-Score: 0.95

The model correctly classified 91 out of 100 resumes into the right job category

 Key Learnings

1.TF-IDF is effective for extracting meaningful features from text data
2.Cosine Similarity works well for matching skill profiles to job requirements
3.Skill gap analysis adds real practical value beyond just prediction

How to Run:

git clone https://github.com/yourusername/ai-career-path-recommendation.git

pip install numpy pandas scikit-learn

python ai_career_path_recommendation.py

Project Structure:

ai-career-path-recommendation/

│

├── ai_career_path_recommendation.py

├── README.md  

 Author:

 Rakcha H

 2nd Year IT Student | SRM Madurai College for Engineering and Technology

 https://www.linkedin.com/in/rakcha-h-792b90394


This project was completed as part of a Machine Learning Internship at Jyesta Corporate Entity (Jan–Mar 2026)

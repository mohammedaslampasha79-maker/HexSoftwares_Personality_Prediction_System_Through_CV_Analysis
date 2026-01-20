# HexSoftwares_Personality_Prediction_System_Through_CV_Analysis
Developed a Personality Prediction System that analyzes resumes using NLP and machine learning. The system extracts text from CVs, predicts key personality traits like leadership, creativity, and analytical thinking, and provides personalized suggestions to improve resume quality and role fit.
# Personality Prediction System Through CV Analysis

This project is a machine learning–based web application that analyzes a candidate’s resume (CV) to predict personality traits. It uses Natural Language Processing (NLP) techniques to extract information from resumes and provides personalized suggestions to improve resume quality.

## Features
- Upload resume in PDF format
- Extracts resume text using PyPDF2
- Applies TF-IDF for text feature extraction
- Predicts personality traits using a Random Forest classifier
- Supports multiple traits including leadership, creativity, teamwork, and analytical thinking
- Provides resume improvement tips based on predicted traits
- Simple and user-friendly Flask web interface

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- PyPDF2
- HTML & CSS

## Project Structure
app.py
cv_data.csv
templates/
├── index.html
└── result.html

## How to Run
1. Clone the repository

2. Navigate to the project directory

3. Install dependencies

4. Run the application

5. Open your browser and go to
http://127.0.0.1:5000/

## Use Case
This system can help recruiters make data-driven hiring decisions and assist candidates in understanding their personality strengths and improving their resumes.


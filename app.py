from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import PyPDF2

app = Flask(__name__)

# 1️⃣ Load dataset
data = pd.read_csv("cv_data.csv")

# 2️⃣ Map descriptive traits to Big Five categories
def map_to_category(description):
    desc = description.lower()
    
    if any(word in desc for word in ["social", "present", "communication", "leadership"]):
        return "extraversion"
    
    elif any(word in desc for word in ["organized", "discipline", "attention", "reliability", "accuracy"]):
        return "conscientiousness"
    
    elif any(word in desc for word in ["creative", "innovative", "curiosity", "explore", "open"]):
        return "openness"
    
    elif any(word in desc for word in ["empathy", "cooperative", "team", "support", "help"]):
        return "agreeableness"
    
    elif any(word in desc for word in ["resilient", "calm", "stress", "pressure", "patience"]):
        return "emotional_stability"
    
    # New traits as separate elif statements
    elif any(word in desc for word in ["lead", "delegat", "decision", "initiative"]):
        return "leadership"
    
    elif any(word in desc for word in ["analyze", "logic", "problem-solving", "reasoning", "data"]):
        return "analytical_thinking"
    
    elif any(word in desc for word in ["creative", "innovation", "original", "idea", "inventive"]):
        return "creativity"
    
    elif any(word in desc for word in ["adapt", "flexible", "adjust", "change", "learning"]):
        return "adaptability"
    
    elif any(word in desc for word in ["teamwork", "collaboration", "cooperate", "group", "peer"]):
        return "teamwork"
    
    elif any(word in desc for word in ["communicate", "presentation", "report", "verbal", "writing"]):
        return "communication_skills"
    
    elif any(word in desc for word in ["problem-solving", "troubleshoot", "challenge", "solution", "resolve"]):
        return "problem_solving"
    
    elif any(word in desc for word in ["innovate", "new method", "optimization", "idea", "novel"]):
        return "innovation"
    
    elif any(word in desc for word in ["time management", "deadline", "multi-task", "schedule", "organize"]):
        return "time_management"
    
    elif any(word in desc for word in ["professional", "ethics", "responsible", "integrity", "accountability"]):
        return "professionalism"
    
    else:
        return "well_rounded"


data['trait_category'] = data['trait_description'].apply(map_to_category)

# 3️⃣ Prepare features and labels
X = data['cv_text']
y = data['trait_category']

# 4️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6️⃣ Train classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_tfidf, y_train)

# 7️⃣ Resume improvement tips for each trait
improvement_tips = {
    "extraversion": "Include more examples of teamwork, leadership, and presentations in your resume to showcase social and communication skills.",
    "conscientiousness": "Add detailed project timelines, achievements, and metrics to show organization, discipline, and reliability.",
    "openness": "Highlight creative projects, innovative solutions, and learning new skills to demonstrate curiosity and adaptability.",
    "agreeableness": "Include volunteer work, collaborations, or mentorship experiences to show empathy and teamwork.",
    "emotional_stability": "Show examples of handling stressful projects, multitasking, or problem-solving under pressure to indicate resilience.",
    "Leadership":"Include project lead roles, initiative-taking, mentoring, and strategic decision-making examples.",
    "Analytical Thinking" : "Highlight data analysis, research projects, problem-solving examples, and measurable outcomes.",
    "Creativity":"Include creative projects, design thinking, prototype development, and inventive approaches in coursework or personal projects.",
    "Adaptability":"Show experiences learning new technologies, shifting roles, or handling changing requirements.",
    "Teamwork":"Highlight group projects, team achievements, cooperative roles, and conflict resolution experiences.",
    "Communication Skills":"Include presentations, reports, public speaking, teaching, or mentoring experiences.",
    "Problem-Solving":"Mention examples of overcoming obstacles, troubleshooting, and successful project execution.",
    "Innovation":"Highlight projects where you developed something new, optimized processes, or used novel solutions.",
    "Time Management":"Include multi-project handling, timely submissions, and prioritization examples.",
    "Professionalism":"Highlight reliability, integrity, project accountability, and professional conduct.",
    "well_rounded":"Shows a balanced mix of skills,adaptability and teamwork. Can contribute effectively in diverse work environments."
}

# 8️⃣ Predict personality and improvement tips
def predict_personality(cv_text):
    vec = vectorizer.transform([cv_text])
    category = model.predict(vec)[0]
    
    # Pick a random 1–2 line description from dataset
    descriptions = data[data['trait_category'] == category]['trait_description'].tolist()
    description = random.choice(descriptions)
    
    # Get improvement tip
    tip = improvement_tips.get(category, "No tips available")
    
    return description, tip

# 9️⃣ Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

#  🔟 Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    cv_text = extract_text_from_pdf(file)
    
    description, tip = predict_personality(cv_text)
    
    return render_template('result.html', personality=description, improvement=tip)

if __name__ == "__main__":
    app.run(debug=True)





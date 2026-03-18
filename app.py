import streamlit as st
import PyPDF2
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# Load NLP Model
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# OpenAI Setup
# -------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# Users (Login)
# -------------------------------
users = {
    "mahesh": "1234",
    "admin": "admin"
}

# -------------------------------
# Session State Init
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------------
# Login Function
# -------------------------------
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")

# -------------------------------
# STOP if not logged in
# -------------------------------
if not st.session_state["logged_in"]:
    login()
    st.stop()

# -------------------------------
# 🎨 Premium UI
# -------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.main-title {
    font-size: 40px;
    text-align: center;
    color: #00c6ff;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚀 AI Resume Analyzer Pro</p>', unsafe_allow_html=True)

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

# -------------------------------
# Functions
# -------------------------------
def extract_text(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

skills_db = [
    "python","java","sql","machine learning","deep learning",
    "nlp","data science","html","css","javascript",
    "react","node","pandas","numpy","power bi","excel"
]

def extract_skills(text):
    doc = nlp(text)
    return list(set([t.text.lower() for t in doc if t.text.lower() in skills_db]))

def ats_score(text, skills):
    score = len(skills) * 8
    if len(text) > 1000:
        score += 10
    for sec in ["education","experience","projects","skills"]:
        if sec in text:
            score += 10
    return min(score, 100)

def section_score(text):
    return {
        "Education": 10 if "education" in text else 0,
        "Experience": 20 if "experience" in text else 0,
        "Projects": 20 if "projects" in text else 0,
        "Skills": 20 if "skills" in text else 0
    }

def job_match(resume, job_desc):
    tfidf = TfidfVectorizer()
    vec = tfidf.fit_transform([resume, job_desc])
    return int(cosine_similarity(vec[0:1], vec[1:2])[0][0]*100)

def keyword_gap(resume, job):
    return list(set(job.split()) - set(resume.split()))[:20]

def ai_feedback(text):
    prompt = f"Analyze this resume and give improvements:\n{text}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

def chatbot(resume, q):
    prompt = f"Resume:\n{resume}\nQuestion:{q}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

def suggestions(score, skills):
    tips=[]
    if score<40: tips.append("Add more skills & projects")
    if len(skills)<5: tips.append("Include more skills")
    if "machine learning" not in skills: tips.append("Add ML projects")
    return tips

def generate_pdf(skills, score, tips):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []
    content.append(Paragraph("Resume Report", styles["Title"]))
    content.append(Paragraph(f"Score: {score}", styles["Normal"]))
    content.append(Paragraph(f"Skills: {', '.join(skills)}", styles["Normal"]))
    for t in tips:
        content.append(Paragraph(t, styles["Normal"]))
    doc.build(content)

# -------------------------------
# MAIN APP
# -------------------------------
if uploaded_file:

    text = extract_text(uploaded_file)
    skills = extract_skills(text)
    score = ats_score(text, skills)

    # Save history
    st.session_state["history"].append({"score":score,"skills":len(skills)})

    # Cards UI
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Skills")
    st.write(skills)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 ATS Score")
    st.progress(score)
    st.write(f"{score}/100")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section Score
    st.subheader("📊 Section Score")
    for k,v in section_score(text).items():
        st.write(f"{k}: {v}")

    # Chart
    if skills:
        fig, ax = plt.subplots()
        ax.barh(skills,[1]*len(skills))
        st.pyplot(fig)

    # Suggestions
    tips = suggestions(score, skills)
    st.subheader("💡 Suggestions")
    for t in tips:
        st.warning(t)

    # Job Matching
    st.subheader("🎯 Job Match")
    job_desc = st.text_area("Paste Job Description")
    if job_desc:
        match = job_match(text, job_desc)
        st.write(f"Match: {match}%")

        st.subheader("⚠️ Missing Keywords")
        st.write(keyword_gap(text, job_desc))

    # AI Feedback
    if st.button("🤖 AI Feedback"):
        st.write(ai_feedback(text))

    # Chatbot
    st.subheader("🤖 Chatbot")
    q = st.text_input("Ask about your resume")
    if q:
        st.write(chatbot(text, q))

    # Dashboard
    st.subheader("📊 Analytics")
    scores = [h["score"] for h in st.session_state["history"]]
    if scores:
        fig, ax = plt.subplots()
        ax.plot(scores)
        st.pyplot(fig)

    # PDF
    generate_pdf(skills, score, tips)
    with open("report.pdf","rb") as f:
        st.download_button("📥 Download Report", f, "report.pdf")

# Logout
if st.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.experimental_rerun()
# app.py
import streamlit as st
from sentence_transformers import util
import numpy as np
import pandas as pd
from math import pi
import os
from datetime import datetime

from data_functions import *
from model_functions import *


# ========== Streamlit Configuration ==========
st.set_page_config(page_title="Job Profile Recommendation", layout="wide")
st.title("Job Profile Recommendation")

try:
    model = load_model()
    # ========== Competences book ==========
    job_profiles,competency_list = load_data()
except Exception as e:
    st.error(f"Can't load SBERT model : {e}")
    st.stop()


# ========== Sidebar ==========
with st.sidebar:
    st.header("Options")
    save_data = st.checkbox("Save answers (CSV)", value=True)
    if save_data :
        csv_name = st.text_input("Name of your csv file",value="answer")
        # Just in case the user put .csv at the end of the name of the file
        if not csv_name.endswith(".csv"):
            csv_name = f"{csv_name}.csv"
        st.write(f"File name: **{csv_name}**")
    st.markdown("---")
    st.text("Group members: Elise Bruneton, Tom Ballet, Lucas Balbi, Alexis Lulin")
    


# ========== Form ==========
with st.form("questionnaire_form"):
    col1, col2 = st.columns(2)

    # left column
    with col1:
        python_level = st.slider("Rate your proficiency in Python", 1, 5, 3)
        ml_level = st.slider("Rate your proficiency in Machine Learning", 1, 5, 3)
        math_level = st.slider("Rate your proficiency in Mathematics", 1, 5, 2)

        lib_used = st.multiselect(
            "Python library you are familiar with",
            ["pandas", "numpy","scikit-learn", "polars", "streamlit", "flask", "matplotlib", "tensorflow", "other"]
        )

        checkbox_nlp = st.checkbox("I have already worked on NLP projects (texts, cats, classification)")
        checkbox_deploy = st.checkbox("I have already deployed a model (API, Docker, cloud)")

    # Right column
    with col2:
        experience_text = st.text_area(
            "Describe your experiences (projects, tasks, technologies) in a few sentences",
            value="Example: I cleaned datasets, built dashboards, trained classification models..."
        )
        projects_text = st.text_area(
            "Give 1-3 project examples (or leave it blank)",
            value=""
        )
        motivation_text = st.text_area("What are your career preferences/areas of interest? (e.g., health, finance, research, etc.)",
                                  value="")

    submit = st.form_submit_button("Analyze my profile")

if not submit: # message
    st.info("Complete the questionnaire then click on *Analyze my profile* to start the analysis.")
    st.stop()



# ========== Prepare data ==========
user_texts = []
if experience_text and experience_text.strip():
    user_texts.append(experience_text.strip())
if projects_text and projects_text.strip():
    user_texts.append(projects_text.strip())
if motivation_text and motivation_text.strip():
    user_texts.append(motivation_text.strip())

# Add informations from sliders and checkboxs for more details
user_texts.append(f"I have Python skill level {python_level}")
user_texts.append(f"I have Machine Learning skill level {ml_level}")
user_texts.append(f"I have Mathematics skill level {math_level}")
if lib_used:
    user_texts.append("I have used: " + ", ".join(lib_used))
if checkbox_nlp:
    user_texts.append("I have worked on NLP projects (text classification, preprocessing)")
if checkbox_deploy:
    user_texts.append("I have deployed models using Docker/Cloud")

# Ensure non-empty text
if len(user_texts) == 0:
    st.error("No user text detected — add free text and try again")
    st.stop()


user_texts = [clean_text(t) for t in user_texts if isinstance(t, str) and t.strip()]


# ========== Embeddings & Scoring (using cosine similarity)==========

competence_texts = competency_list["CompetencyName"].tolist()
user_text = " ".join(user_texts)

competence_embeddings = model.encode(competence_texts, convert_to_tensor=True)
user_embedding = model.encode(user_text, convert_to_tensor=True)

cosine_scores = util.cos_sim(user_embedding, competence_embeddings)[0].cpu().numpy()

competency_list["similarity"] = cosine_scores


# ========== Scoring per bloc / job ==========
block_scores = {}
block_details = {}
coverage_scores = {}

coverage_threshold = 0.2 

for _, row in job_profiles.iterrows():
    job = row["Job"]
    comp_ids = [c.strip() for c in row["Competencies"].split(",") if c.strip()]
    
    comp_sim_values = competency_list.loc[
        competency_list["CompID"].isin(comp_ids), "similarity"
    ].values
    
    if len(comp_sim_values) > 0:
        score = np.mean(comp_sim_values)
        block_scores[job] = score

        covered = np.sum(comp_sim_values >= coverage_threshold)
        coverage = covered / len(comp_sim_values)
        coverage_scores[job] = coverage

        block_details[job] = {
            "phrases": competency_list.loc[competency_list["CompID"].isin(comp_ids), "CompetencyName"].tolist(),
            "max_similarities": comp_sim_values.tolist(),
        }


# ========== Global score ==========
global_score = float(np.mean(list(block_scores.values())))

# ========== Work Recommendation ==========
job_scores = block_scores.copy()
sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

# We select all jobs that scored better than average (or the 5 best ones if all below average)
threshold = global_score
recommended = [(job, s) for job, s in sorted_jobs if s >= threshold]

if not recommended:  # Case where no job is above the threshold
    recommended = sorted_jobs[:5]




# ========== Results ==========
#Display a visualization of the results 
st.header("Analysis results")

# Bar chart
scores_df = pd.DataFrame({
    "similarity_score": block_scores,
    "coverage_score": coverage_scores
}).sort_values("similarity_score", ascending=False)

st.subheader("Scores of 5 first")
st.dataframe(scores_df[:5].style.format("{:.3f}"))

# Bar chart pour le score de similarité
st.bar_chart(scores_df["similarity_score"])

# Global score
st.subheader("Global score")
st.metric("global score",f"{global_score:.3f}")


with st.expander("See the details of similarities by sentence (explainability)"):
    # Sorts blocks by score, descending order to get the best results first
    sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)

    for block, score in sorted_blocks:
        det = block_details[block]
        st.markdown(f"**Bloc : {block} — score {score:.3f}**")
        df = pd.DataFrame({
            "competence_phrase": det["phrases"],
            "max_similarity": det["max_similarities"]
        }).sort_values("max_similarity", ascending=False)
        st.table(df.style.format({"max_similarity": "{:.3f}"}))


radar_fig = make_radar_chart(block_scores)
st.pyplot(radar_fig)

# All jobs
st.subheader("All scores")
st.dataframe(scores_df.style.format("{:.3f}"))

# ========== Saving answer in csv ==========
if save_data:
    out_dir = "answers" # directory for answers
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir, csv_name) # path for file

    record = {
        "timestamp": timestamp,
        "python_level": python_level,
        "ml_level": ml_level,
        "nlp_level": math_level,
        "tools_used": "|".join(lib_used),
        "checkbox_nlp": checkbox_nlp,
        "checkbox_deploy": checkbox_deploy,
        "experience_text": experience_text,
        "projects_text": projects_text,
        "motivation": motivation_text,
        "global_score": global_score,
        "top_jobs": "|".join([f"{j}:{job_scores[j]:.3f}" for j in job_scores])
    }
    pd.DataFrame([record]).to_csv(filename, index=False) # save results in a csv
    
    with st.sidebar: # message in sidebar
        st.markdown("---")
        st.success(f"Answer saved in `{filename}`")

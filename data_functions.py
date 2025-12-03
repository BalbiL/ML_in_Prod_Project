import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def convert_to_ids(comp_string, comp_dict):
        comps = [c.strip() for c in re.split(r'[,;|]', comp_string) if c.strip()]
        ids = [comp_dict.get(c, None) for c in comps if c in comp_dict]
        return ', '.join(ids)

@st.cache_resource
def load_data():
    # Function to load and clean data, returning job profiles and competency dataframes ready for use
    # load
    S3_BUCKET_PATH = "s3://goup2-data/job_dataset.csv"
    try:
        df = pd.read_csv(S3_BUCKET_PATH, storage_options={"anon": True})
    except Exception as e:
        st.error(f"Erreur S3 : {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Clean
    df = df.drop(columns=["ExperienceLevel","Responsibilities", "YearsOfExperience", "Keywords"])
    df = df.rename(columns={"Skills": "Competencies","Title":"Job"})

    df["Job"] = df["Job"].str.lower().str.strip()

    df["Job"] = df["Job"].str.replace(r"machine[\s\-_]*learning", "ml", regex=True)
    df["Job"] = df["Job"].str.replace(r"deep[\s\-_]*learning", "dl", regex=True)
    df["Job"] = df["Job"].str.replace(r"data[\s\-_]*scientist", "data scientist", regex=True)
    df["Job"] = df["Job"].str.replace(r"\bengineer\b", "eng", regex=True)
    df["Job"] = df["Job"].str.replace(r"\bdeveloper\b", "dev", regex=True)
    df["Job"] = df["Job"].str.replace(r"\bscientist\b", "science", regex=True)

    df["Job"] = df["Job"].str.replace(
        r"(junior|jr\.?|entry[\s\-_]*level|intern|fresher|trainee|assistant|associate|experienced|senior|lead|principal|head|specialist)",
        "", regex=True
    )

    df["Job"] = df["Job"].str.replace(r"[-_/]", " ", regex=True)
    df["Job"] = df["Job"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["Job"] = df["Job"].str.title()

    df_job = (
        df.groupby("Job", as_index=False)
        .agg({"Competencies": lambda x: ', '.join(set(', '.join(x.dropna()).split(',')))})
    )
    df_job["Competencies"] = df_job["Competencies"].apply(
        lambda s: ', '.join(sorted({c.strip() for c in s.split(',') if c.strip()}))
    )

    # Extract competencies
    all_competencies = []
    for comps in df_job["Competencies"].dropna():
        split_comps = re.split(r'[,;|]', comps)
        cleaned = [c.strip() for c in split_comps if c.strip()]
        all_competencies.extend(cleaned)

    unique_comps = sorted(set(all_competencies))

    # Dataframe for competencies
    df_comp = pd.DataFrame({
        "CompID": [f"C{str(i+1).zfill(3)}" for i in range(len(unique_comps))],
        "CompetencyName": unique_comps
    })

    # replace name per iD
    comp_dict = dict(zip(df_comp["CompetencyName"], df_comp["CompID"]))

    df_job["Competencies"] = df_job["Competencies"].apply(lambda x: convert_to_ids(x, comp_dict))

    return df_job,df_comp




# ========== Clean data ==========
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(lemmas)

# user_texts = [clean_text(t) for t in user_texts if isinstance(t, str) and t.strip()]


# ========== Radar chart ==========
#Display the result on a radar chart for visualization
def make_radar_chart(scores_dict, title="Skills"):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    N = len(labels)

    # close the plot
    values += values[:1]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], labels)

    # Draw ylabels
    ax.set_rlabel_position(0)
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    plt.title(title)
    return fig
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime as dt
from pathlib import Path

# -----------------------------
# Load trained pipeline
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model" / "xgb_final_model.joblib"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Course Completion Prediction",
    layout="centered"
)

st.title("Student Course Completion Prediction")
st.write(
    """
    This application predicts whether a student is likely to **complete**
    an online course based on engagement, demographics, and course behavior.
    """
)

st.markdown("---")
st.subheader("Enter Student Information")

# -----------------------------
# INPUTS (RAW FEATURES ONLY)
# -----------------------------

# Course
course_id = st.selectbox(
    "Course ID",
    [
        "HarvardX/CB22x/2013_Spring",
        "HarvardX/CS50x/2012",
        "HarvardX/ER22x/2013_Spring",
        "HarvardX/PH207x/2012_Fall",
        "HarvardX/PH278x/2013_Spring",
        "MITx/6.002x/2012_Fall",
        "MITx/6.002x/2013_Spring",
        "MITx/14.73x/2013_Spring",
        "MITx/2.01x/2013_Spring",
        "MITx/3.091x/2012_Fall",
        "MITx/3.091x/2013_Spring",
        "MITx/6.00x/2012_Fall",
        "MITx/7.00x/2013_Spring",
        "MITx/8.02x/2013_Spring",
        "MITx/6.00x/2013_Spring",
        "MITx/8.MReV/2013_Summer",
    ]
)

# Engagement flags
viewed = st.selectbox("Viewed Course Content?", [0, 1])
explored = st.selectbox("Explored Course?", [0, 1])

# Country
final_cc_cname_DI = st.selectbox(
    "Country",
    [
        "United States", "France", "Unknown/Other", "Mexico", "Australia",
        "India", "Canada", "Other South Asia",
        "Other North & Central Amer., Caribbean", "Other Oceania",
        "Japan", "Other Africa", "Colombia", "Russian Federation",
        "Other Europe", "Germany", "Other Middle East/Central Asia",
        "Poland", "Indonesia", "Bangladesh", "China",
        "United Kingdom", "Spain", "Ukraine", "Greece",
        "Pakistan", "Nigeria", "Egypt", "Other South America",
        "Brazil", "Portugal", "Other East Asia",
        "Philippines", "Morocco"
    ]
)

# Education level
LoE_DI = st.selectbox(
    "Level of Education",
    ["Unknown", 'Secondary', "Master's", "Bachelor's", 'Doctorate',
       'Less than Secondary']
)
LoE_DI = np.nan if LoE_DI == "Unknown" else LoE_DI

# Year of Birth
YoB = st.number_input(
    "Year of Birth (YoB)",
    min_value=1900,
    max_value=2025,
    value=1995
)

# Gender
gender = st.selectbox(
    "Gender",
    ["m", "f", "o", "Unknown"]
)
gender = np.nan if gender == "Unknown" else gender

# Dates
start_time_DI = st.date_input(
    "Course Start Date",
    min_value=dt.date(2012, 1, 1),
    max_value=dt.date(2013, 12, 31),
    value=dt.date(2013, 1, 1)
)

last_event_DI = st.date_input(
    "Last Activity Date",
    min_value=dt.date(2012, 1, 1),
    max_value=dt.date(2013, 12, 31),
    value=dt.date(2013, 6, 1)
)
st.caption("Dates are restricted to the course years present in the training data (2012–2013).")

# Engagement metrics
nevents = st.number_input("Number of Events", min_value=0.0, value=5.0)
ndays_act = st.number_input("Active Days", min_value=0.0, value=2.0)
nchapters = st.number_input("Chapters Accessed", min_value=0.0, value=1.0)
nforum_posts = st.number_input("Forum Posts", min_value=0, value=0)

# Historical behavior
no_of_courses_registered = st.number_input(
    "Number of Courses Registered (Historical)",
    min_value=0,
    value=1
)

no_of_courses_explored = st.number_input(
    "Number of Courses Explored (Historical)",
    min_value=0,
    value=1
)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_df = pd.DataFrame({
    "course_id": [course_id],
    "viewed": [viewed],
    "explored": [explored],
    "final_cc_cname_DI": [final_cc_cname_DI],
    "LoE_DI": [LoE_DI],
    "YoB": [float(YoB)],
    "gender": [gender],
    "start_time_DI": [str(start_time_DI)],
    "last_event_DI": [str(last_event_DI)],
    "nevents": [float(nevents)],
    "ndays_act": [float(ndays_act)],
    "nchapters": [float(nchapters)],
    "nforum_posts": [int(nforum_posts)],
    "no_of_courses_registered": [int(no_of_courses_registered)],
    "no_of_courses_explored": [int(no_of_courses_explored)]
})

# -----------------------------
# Prediction
# -----------------------------
THRESHOLD = 0.89

if st.button("Predict Completion"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = "COMPLETE" if prob > THRESHOLD else "NOT COMPLETE"

    st.markdown("---")
    st.subheader("Prediction Result")

    st.write(f"**Completion Probability:** {prob:.2f}")
    st.write(f"**Decision Threshold:** {THRESHOLD}")

    if prediction == "COMPLETE":
        st.success("The student is likely to COMPLETE the course.")
    else:
        st.error("The student is likely to NOT COMPLETE the course.")

    st.caption(
        "Prediction is based on an XGBoost model trained with engineered "
        "engagement and behavioral features."
    )

    st.markdown("### Why this prediction was made")

    if prediction == "COMPLETE":
        st.success(
            "This student shows strong engagement with the course. "
            "High activity levels, multiple active days, and consistent interaction "
            "with course content significantly increase the likelihood of completion."
        )
    else:
        st.warning(
            "This student shows limited engagement with the course. "
            "Lower activity levels and fewer active days reduce the likelihood "
            "of course completion."
        )

    st.markdown(
        "**Key factors considered:**\n"
        "- Course activity (number of events)\n"
        "- Number of active days\n"
        "- Chapters accessed\n"
        "- Course exploration behavior"
    )
    
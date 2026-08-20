import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", page_icon="📚")
st.title("📚 Student Performance Predictor")
st.write("Enter student information to estimate the final score.")

model = joblib.load("student_performance_model.pkl")

study_hours = st.slider("Study hours per day", 0.0, 12.0, 5.0, 0.5)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
previous_score = st.slider("Previous score (%)", 0.0, 100.0, 70.0, 1.0)
assignments = st.slider("Assignments completed", 0, 10, 8)
sleep_hours = st.slider("Sleep hours per day", 0.0, 12.0, 7.0, 0.5)

if st.button("Predict Performance"):
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percent": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments,
        "sleep_hours": sleep_hours
    }])
    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))
    st.success(f"Estimated final score: {prediction:.1f}%")
    if prediction >= 75:
        st.info("Performance category: Strong")
    elif prediction >= 50:
        st.info("Performance category: Moderate")
    else:
        st.warning("Performance category: Needs improvement")

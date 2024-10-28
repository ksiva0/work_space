import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_excel(r"C:\Users\sivak\Excel_files\Assessment_Data.xlsx")

# Prepare data by creating a unique identifier for each student
data['studentID'] = data['firstName'] + ' ' + data['lastName']
pivot_table = data.pivot_table(index='studentID', columns='assessmentTitle', values='overallScore', fill_value=0)
students = pivot_table.index

# Collaborative Filtering (SVD)
svd = TruncatedSVD(n_components=10, random_state=42)
latent_matrix = svd.fit_transform(pivot_table)
student_similarity = cosine_similarity(latent_matrix)

# Neural Collaborative Filtering (NCF) Model Preparation
data['student_idx'] = data['studentID'].astype("category").cat.codes
data['assessment_idx'] = data['assessmentTitle'].astype("category").cat.codes
X = data[['student_idx', 'assessment_idx']]
y = data['overallScore']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_students = data['student_idx'].nunique()
num_assessments = data['assessment_idx'].nunique()

class NCFModel(tf.keras.Model):
    def __init__(self, num_students, num_assessments, latent_dim=10):
        super(NCFModel, self).__init__()
        self.student_embedding = tf.keras.layers.Embedding(num_students, latent_dim)
        self.assessment_embedding = tf.keras.layers.Embedding(num_assessments, latent_dim)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        student_idx, assessment_idx = inputs[:, 0], inputs[:, 1]
        student_vec = self.student_embedding(student_idx)
        assessment_vec = self.assessment_embedding(assessment_idx)
        concat = tf.concat([student_vec, assessment_vec], axis=-1)
        return self.dense(concat)

ncf_model = NCFModel(num_students, num_assessments)
ncf_model.compile(optimizer='adam', loss='mean_squared_error')
ncf_model.fit(X_train.values, y_train, validation_data=(X_val.values, y_val), epochs=5, batch_size=32)

# Define recommendation functions
def get_collaborative_recommendations(student_id, top_n=5):
    student_idx = np.where(students == student_id)[0][0]
    sim_scores = student_similarity[student_idx]
    similar_students_idx = np.argsort(sim_scores)[::-1][1:top_n + 1]
    similar_students = pivot_table.iloc[similar_students_idx]
    completed_assessments = pivot_table.columns[pivot_table.iloc[student_idx] > 0]
    max_scores = similar_students.max()
    recommendations = max_scores.drop(completed_assessments).sort_values(ascending=False)
    return recommendations.head(top_n).index.tolist(), recommendations.head(top_n).values

def get_ncf_recommendations(student_id, top_n=5):
    student_idx = data[data['studentID'] == student_id]['student_idx'].iloc[0]
    uncompleted_assessments = pivot_table.columns[pivot_table.loc[student_id] == 0]
    assessment_idxs = data[data['assessmentTitle'].isin(uncompleted_assessments)]['assessment_idx'].unique()
    student_inputs = np.array([[student_idx, assessment_idx] for assessment_idx in assessment_idxs])
    predicted_scores = ncf_model.predict(student_inputs).flatten()
    recommendations = sorted(zip(uncompleted_assessments, predicted_scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [assessment for assessment, score in recommendations]

# Streamlit application setup
st.title("Student Assessment Recommendation System")
st.write("Choose a recommendation method and see top assessment recommendations for a student.")

selected_student = st.selectbox("Select a student", students)
top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
method = st.radio("Select recommendation method", ('Collaborative Filtering', 'Neural Collaborative Filtering'))

if st.button("Get Recommendations"):
    if method == 'Collaborative Filtering':
        assessment_titles, scores = get_collaborative_recommendations(selected_student, top_n=top_n)
        st.write(f"### Collaborative Filtering Recommendations for {selected_student}:")
        for title, score in zip(assessment_titles, scores):
            st.write(f"- {title} (Score: {score:.2f})")
    elif method == 'Neural Collaborative Filtering':
        assessment_titles = get_ncf_recommendations(selected_student, top_n=top_n)
        st.write(f"### NCF Recommendations for {selected_student}:")
        for title in assessment_titles:
            st.write(f"- {title}")

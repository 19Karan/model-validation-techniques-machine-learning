import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(page_title="Model Validation Techniques", layout="centered")
st.title("📊 Model Validation Techniques Comparison")

df = pd.read_csv("data/pima_indians_diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = DecisionTreeClassifier(random_state=42)

method = st.selectbox(
    "Select Validation Technique",
    ["Train-Test Split", "K-Fold", "Stratified K-Fold", "Leave-One-Out"]
)

if method == "Train-Test Split":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

elif method == "K-Fold":
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    st.write("Cross Validation Scores:", scores)
    st.write("Mean Accuracy:", np.mean(scores))

elif method == "Stratified K-Fold":
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf)
    st.write("Stratified CV Scores:", scores)
    st.write("Mean Accuracy:", np.mean(scores))

else:
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo)
    st.write("Leave-One-Out Mean Accuracy:", np.mean(scores))

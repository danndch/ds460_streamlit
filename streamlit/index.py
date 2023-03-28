import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA # feature reduction method

#https://www.youtube.com/watch?v=Klqn--Mu2pE&t=211s

st.title("Streamlit Example")

st.write("""
# Explore Different Classifier
Which one is the best?
""")

# These are the sidebars
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))


# -----------------------------------------------------------------------------------------------------
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

# -----------------------------------------------------------------------------------------------------
# Creation of sliders
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max_Depth", 2, 15)
        n_estimators = st.sidebar.slider("Number_Estimators", 1, 100)
        params["Max_Depth"] = max_depth
        params["Number_Estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

# -----------------------------------------------------------------------------------------------------
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["Number_Estimators"],
                                     max_depth = params["Max_Depth"], random_state = 1234)
    return clf


clf = get_classifier(classifier_name, params)

# -----------------------------------------------------------------------------------------------------
# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# -----------------------------------------------------------------------------------------------------
# Plot 
pca = PCA(2)
X_projected = pca.fit_transform(X)


x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show
st.pyplot(fig)







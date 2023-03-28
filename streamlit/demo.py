import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import pandas as pd
from PIL import Image

st.title('setsosa, versicolour, or virginica ?')

image = Image.open(r'C:\Users\carlos\Desktop\iris.jpg')

st.image(image)

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
params = add_parameter_ui(classifier_name)
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf
clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

st.sidebar.title("Attributes")
st.sidebar.write("Choose the attributes related to your plant: ")

sepal_length = st.sidebar.slider("Choose sepal length:", min_value = 4.0, max_value = 8.0, step=0.1)
sepal_width = st.sidebar.slider("Choose sepal width:", min_value = 2.0, max_value = 4.5, step=0.1)
petal_length = st.sidebar.slider("Choose petal length:", min_value = 1.0, max_value = 7.0, step=0.1)
petal_width = st.sidebar.slider("Choose petal length:", min_value = 0.0, max_value = 2.6, step=0.1)

user_input=pd.DataFrame({'sepal length (cm)':[sepal_length], 'sepal width (cm)':[sepal_width], 'petal length (cm)':[petal_length], 'petal width (cm)':[petal_width]})

predict = st.sidebar.button("Predict")
if predict:
    pred = clf.predict(user_input)
    st.write(pred)

st.write("""
## Prediction""")


if pred[0] == 0:
    st.success("you have iris-setosa ")

if pred[0] == 1:
    st.success("you have iris-versicolor ")

if pred[0] == 2:
    st.success("you have iris-virginica ")


st.write("""
## Model Details""")

col1, col2 = st.columns([1, 1])

with col1: 
    plt.figure()
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
    st.pyplot(fig)

with col2:
    plt.figure(figsize = (10,6))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    explainer = shap.Explainer(clf.predict, X_train)
    shap_values_ttl = explainer(X_test)
    fig_ttl = shap.plots.beeswarm(shap_values_ttl)
    st.pyplot(fig_ttl)
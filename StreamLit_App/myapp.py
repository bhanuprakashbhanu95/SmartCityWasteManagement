import streamlit as st
from sklearn import svm
from sklearn import datasets
import pickle

clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)


st.download_button(
    "Download Model",
    data=pickle.dumps(clf),
    file_name="model.pkl",
)

uploaded_file = st.file_uploader("Upload Model")

if uploaded_file is not None:
    clf2 = pickle.loads(uploaded_file.read())
    st.write("Model loaded")
    st.write(clf2)
    st.write("Predicting...")
    st.write(clf2.predict(X[0:1]))
    st.write(y[0])
    st.write("Done!")

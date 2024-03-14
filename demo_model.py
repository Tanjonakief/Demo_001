import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets 

iris = datasets.load_iris()
X = iris.data
y = iris.target

model_rfc = RandomForestClassifier()
model_rfc.fit(X,y)


st.title('Model ML')
sepal_length = st.sidebar.slider(label='Sepal_length', min_value=0.0, max_value=10.0, value=0.0)
sepal_width = st.sidebar.slider(label='Sepal_width', min_value=0.0, max_value=10.0, value=0.0)
petal_length = st.sidebar.slider(label='Petal_length', min_value=0.0, max_value=10.0, value=0.0)
petal_width = st.sidebar.slider(label='Petal_width', min_value=0.0, max_value=10.0, value=0.0)

data = {
    'sepal_length' : sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width
}

df = pd.DataFrame(data, index = [0])
st.write(df)

y_pred = model_rfc.predict(df)
y_pred_proba = model_rfc.predict_proba(df)

y_name = iris.target_names[y_pred[0]]
st.subheader('Prédiction: ')
st.write(y_name)

st.subheader('Prédiction probability: ')
st.write(y_pred_proba)

#print(type(y_name))
#  <a href="https://docs.streamlit.io/library/cheatsheet">Streamlit Cheatsheet</a>

## Command line or terminal
### Install
pip install streamlit

### Import 
import streamlit as st

### Run the app

streamlit run app.py


## Writting in the app

### st.write vs magic commands

 st.write() is a streamlit command that is used to display text, data frames, images, and other types of content within a streamlit app. It is used to display content within the streamlit app, which is different from executing specific tasks.

Therefore, when building a Streamlit app, you would generally use st.write() to display content within the app, while magic commands are used for specific tasks.

## Display data
st.dataframe(my_dataframe) <\br>
st.table(data.iloc[0:10])
st.json({'was':'I','me':'lee'})
st.metric('My metric', 42, 2)


## Display media
st.image('./header.png')
st.audio(data)
st.video(data)


## Display interactive widgets
st.button('Click me')
st.experimental_data_editor('Edit data', data)
st.checkbox('I agree')

## Optimize performance



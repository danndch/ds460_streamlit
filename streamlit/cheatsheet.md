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

So when building a Streamlit app, you would generally use st.write() to display content within the app, while magic commands are used for specific tasks.


## Display text
st.text('Fixed width text') <br/>
st.markdown('_Markdown_') <br/>
st.latex(r''' e^{i\pi} + 1 = 0 ''') <br/>
st.write('Most objects') # df, err, func, keras! <br/>
st.write(['st', 'is <', 3]) <br/>
st.title('My title')


## Display data
st.dataframe(my_dataframe) <br />
st.table(data.iloc[0:10])<br />
st.json({'was':'I','me':'lee'})<br />
st.metric('My metric', 42, 2)<br />


## Display media
st.image('./header.png')<br />
st.audio(data)<br />
st.video(data)<br />


## Display interactive widgets
st.button('Click me')<br />
st.experimental_data_editor('Edit data', data)<br />
st.checkbox('I agree')<br />

## Optimize performance



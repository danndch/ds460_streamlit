# ds460_nuggies_streamlit
Repository for Data Science 460 from BYU-Idaho, helping them set up an machine learning app with Docker.


#Streamlit Application Overview & Docuementation
## What is streamlit

- importable package designed for creating webpages from simple scripts
- uses basic commands to deploy a local webpage with interactive components and other elements from your python code
- this allows us to create our ML model using Python scripts and then deploy it to a shareable webpage


### ML Deployment with Streamlit
App development Example
- https://www.youtube.com/watch?v=Klqn--Mu2pE&t=211s

## App Deployment

Open your terminal
- Run to install streamlit:
```
pip install streamlit
```

Import streamlit
```
import streamlit as st
```

Lets put a simple title
```
st.title("I know what I'm doing")
```

Save the file

In terminal:
```
streamlit run [yourFile].py
```

You're doing GREAT! Lets add somemore stuff!


### Charts in Streamlit

Streamlit documentation and programming examples can be found [here](https://docs.streamlit.io/) 

**Line Charts** 
[Line Chart Documentation](https://docs.streamlit.io/library/api-reference/charts/st.line_chart)

```
import streamlit as st
import pandas as pd
import numpy as np

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)
```

**Bar Charts** 
[Bar Chart Documentation](https://docs.streamlit.io/library/api-reference/charts/st.bar_chart)

```
import streamlit as st
import pandas as pd
import numpy as np

chart_data = pd.DataFrame(
    np.random.randn(50, 3),
    columns=["a", "b", "c"])

st.bar_chart(chart_data)
```


**Maps**
[Map Documentation](https://docs.streamlit.io/library/api-reference/charts/st.map)

```
import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(df)
```


## Links
https://docs.streamlit.io/
https://docs.streamlit.io/library/api-reference/charts
https://docs.streamlit.io/library/api-reference/data
https://docs.streamlit.io/library/api-reference/widgets
https://docs.streamlit.io/library/api-reference/status

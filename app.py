import os
import wandb
import streamlit as st
import streamlit.components.v1 as components 

from utils import train

project = "st"
entity = "capecape"

HEIGHT = 720

def get_project(api, name, entity=None):
    return api.project(name, entity=entity).to_html(height=HEIGHT)

st.title("Let's log some metrics to wandb 👇")

# Sidebar
sb = st.sidebar
sb.title("Train your model")
# wandb_token = sb.text_input("paste your wandb Api key if you want: https://wandb.ai/authorize", type="password")


# wandb.login(key=wandb_token)
wandb.login(anonymous="allow")
api = wandb.Api()

# render wandb dashboard
components.html(get_project(api, project, entity), height=HEIGHT)

# run params
runs = sb.number_input('Number of runs:', min_value=1, max_value=10, value=1)
epochs = sb.number_input('Number of epochs:', min_value=1, max_value=1000, value=100)



pseudo_code = """
We will execute a simple training loop
```python
wandb.init(project="st", ...)
for i in range(epochs):
acc = 1 - 2 ** -i - random()
loss = 2 ** -i + random()
wandb.log({"acc": acc, 
            "loss": loss})
```
"""

sb.write(pseudo_code)

# train model
if sb.button("Run Example"):
    my_bar = sb.progress(0)
    print("Running training")
    for i in range(runs):
        train(project=project, entity=entity, epochs=epochs)
        my_bar.progress((i+1)/runs)
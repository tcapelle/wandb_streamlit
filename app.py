import os, random
import wandb
import streamlit as st
import streamlit.components.v1 as components 

from utils import train, WORDS

project = "st"
entity = "capecape"

HEIGHT = 720

def get_project(api, name, entity=None):
    return api.project(name, entity=entity).to_html(height=HEIGHT)

st.title("The wandb Dashboard 👇")

run_name = "-".join(random.choices(WORDS, k=2)) + f"-{random.randint(0,100)}"

# Sidebar
sb = st.sidebar
sb.title("Train your model")
# wandb_token = sb.text_input("paste your wandb Api key if you want: https://wandb.ai/authorize", type="password")


# wandb.login(key=wandb_token)
wandb.login(anonymous="must")
api = wandb.Api()

st.success(f"You should see a new run named **{run_name}**, it\'ll have a green circle while it\'s still active")


# render wandb dashboard
components.html(get_project(api, project, entity), height=HEIGHT)

# run params
runs = 1
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
    
    print("Running training")
    for i in range(runs):
        my_bar = sb.progress(0)
        train(name=run_name, project=project, entity=entity, epochs=epochs, bar=my_bar)
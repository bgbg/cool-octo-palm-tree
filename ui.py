import streamlit as st
import random
import os

try:
    secrets_dict = dict(st.secrets)
    if not secrets_dict:
        raise Exception
except Exception:
    # Fallback: load from .env if present
    from dotenv import load_dotenv

    load_dotenv()
    secrets_dict = {k: v for k, v in os.environ.items() if not k.startswith("_")}

# Session state for order
if "order" not in st.session_state:
    st.session_state["order"] = list(secrets_dict.keys())

# Filtering option
filter_text = st.text_input("Filter keys")

# Button to randomize order
if st.button("Randomize Order"):
    random.shuffle(st.session_state["order"])

st.write("**Secrets Keys and First 10 Characters of Values:**")
for key in st.session_state["order"]:
    if filter_text and filter_text.lower() not in key.lower():
        continue
    value = str(secrets_dict[key])
    st.write(f"{key}: {value[:10]}...")

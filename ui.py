import streamlit as st
import random
import os
import streamlit_authenticator as stauth

try:
    secrets_dict = dict(st.secrets)
    if not secrets_dict:
        raise Exception
except Exception:
    # Fallback: load from .env if present
    from dotenv import load_dotenv

    load_dotenv()
    secrets_dict = {k: v for k, v in os.environ.items() if not k.startswith("_")}

# Load AUTH_SALT
AUTH_SALT = secrets_dict.get("AUTH_SALT") or os.getenv("AUTH_SALT")
assert AUTH_SALT is not None, "AUTH_SALT is not set"

# Build users from *_HASHED_PW
usernames = {}
for k, v in secrets_dict.items():
    if k.endswith("_HASHED_PW"):
        uname = k[:-10].lower()
        usernames[uname] = {
            "email": f"{uname}@example.com",
            "name": uname.capitalize(),
            "password": v,
        }
assert usernames, "No user hashes found in secrets/env"

config = {
    "credentials": {"usernames": usernames},
    "cookie": {
        "expiry_days": 7,
        "key": AUTH_SALT,
        "name": "demo_auth",
    },
    "preauthorized": {"emails": []},
}

# Custom login form
st.write("## Login")
input_username = st.text_input("Username").strip().lower()
input_password = st.text_input("Password", type="password")
login_btn = st.button("Login")

if "auth_status" not in st.session_state:
    st.session_state["auth_status"] = None
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

if login_btn:
    if input_username in usernames:
        # Use streamlit-authenticator's check_pw
        hashed_pw = usernames[input_username]["password"]
        if stauth.Hasher.check_pw(input_password, hashed_pw):
            st.session_state["auth_status"] = True
            st.session_state["auth_user"] = input_username
        else:
            st.session_state["auth_status"] = False
            st.session_state["auth_user"] = None
    else:
        st.session_state["auth_status"] = False
        st.session_state["auth_user"] = None

if st.session_state["auth_status"]:
    st.write(f'Welcome *{st.session_state["auth_user"]}*')
    if "order" not in st.session_state:
        st.session_state["order"] = list(secrets_dict.keys())
    filter_text = st.text_input("Filter keys")
    if st.button("Randomize Order"):
        random.shuffle(st.session_state["order"])
    st.write("**Secrets Keys and First 10 Characters of Values:**")
    for key in st.session_state["order"]:
        if filter_text and filter_text.lower() not in key.lower():
            continue
        value = str(secrets_dict[key])
        st.write(f"{key}: {value[:10]}...")
elif st.session_state["auth_status"] is False:
    st.error("Invalid username or password")

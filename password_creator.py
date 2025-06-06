import getpass
import os
import streamlit_authenticator as stauth
import pyperclip
import dotenv

dotenv.load_dotenv()
AUTH_SALT = os.getenv("AUTH_SALT", None)
assert AUTH_SALT is not None, "AUTH_SALT is not set"

username = input("Enter username: ")
password = getpass.getpass("Enter password: ")

# Prepare credentials dict for hashing
credentials = {"usernames": {username: {"password": password}}}
hashed_credentials = stauth.Hasher.hash_passwords(credentials)
hashed_pw = hashed_credentials["usernames"][username]["password"]

print(f"Username: {username}")
print(f"Hashed password: {hashed_pw}")

try:
    pyperclip.copy(hashed_pw)
    print("Hashed password copied to clipboard.")
except Exception as e:
    print(f"Could not copy to clipboard: {e}")

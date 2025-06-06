import streamlit as st
import random
import os
import streamlit_authenticator as stauth

st.set_page_config(page_title="RAG QA + Fragments", layout="centered")

# --- Document count debug block ---
from sqlalchemy.exc import SQLAlchemyError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def get_document_count():
    try:
        # Load env if needed
        if load_dotenv:
            load_dotenv()
        connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
        if not connection_string:
            return None, "No SUPABASE_CONNECTION_STRING"
        from orm import get_db_engine, get_db_session, Document

        engine = get_db_engine(connection_string)
        session = get_db_session(engine)
        count = session.query(Document).count()
        session.close()
        return count, None
    except SQLAlchemyError as e:
        return None, f"DB error: {e}"
    except Exception as e:
        return None, str(e)


doc_count, doc_count_err = get_document_count()
if doc_count is not None:
    str_connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
    str_connection_string = (
        str_connection_string[:10] + "..." + str_connection_string[-10:]
    )
    st.info(f"We have {doc_count} documents in {str_connection_string}")
elif doc_count_err:
    st.warning(f"Document count error: {doc_count_err}")
# --- End document count debug block ---

try:
    secrets_dict = dict(st.secrets)
    if not secrets_dict:
        raise Exception
except Exception:
    # Fallback: load from .env if present
    from dotenv import load_dotenv

    load_dotenv()
    secrets_dict = {k: v for k, v in os.environ.items() if not k.startswith("_")}

required_secrets = [
    "AUTH_SALT",
    "SUPABASE_CONNECTION_STRING",
]
for r in required_secrets:
    assert r in secrets_dict, f"{r} is not set"


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
    # --- BEGIN RAG QA UI (from tmp.py) ---
    import pandas as pd
    import time
    import concurrent.futures
    import datetime
    import threading
    import signal
    from search import get_answer, retrieve_documents

    # Patch signal.signal for subthreads (deepeval/Streamlit hack)
    if not threading.current_thread() is threading.main_thread():
        signal.signal = lambda *a, **k: None

    # Set RTL layout for Hebrew
    st.markdown(
        """
        <style>
        body, .main, .stApp {
            direction: rtl;
            text-align: right;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > textarea {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RAG QA: Answer + Fragment Info")

    DEFAULT_QUESTION = "מתי נדרשת בדיקת המטרה על חזיתות?"

    if "auto_run" not in st.session_state:
        st.session_state.auto_run = True

    with st.form("qa_form"):
        question = st.text_area(
            "שאלה (עברית בלבד)",
            DEFAULT_QUESTION,
            height=70,
            key="question",
            help="לחץ Ctrl+Enter לשליחה, Enter לרדת שורה",
        )
        top_k = st.number_input(
            "Top K Fragments", min_value=1, max_value=20, value=10, key="top_k"
        )
        llm_model = st.text_input("LLM Model", value="gpt-4o", key="llm_model")
        submit = st.form_submit_button("שלח שאלה והצג תשובה + פרטי קטעים")

    # Auto-submit on first load
    if st.session_state.auto_run:
        submit = True
        st.session_state.auto_run = False

    log_path = "takanot_rag.log"
    log_container = st.container()
    with log_container:
        log_expander = st.expander("הצג לוג ריצה (takanot_rag.log)", expanded=True)
        log_area = log_expander.empty()
    answer_container = st.empty()

    if submit and question.strip():
        # Clear previous log and answer
        log_area.empty()
        answer_container.empty()
        log_lines = []
        result = None
        # Record log start position
        log_start_pos = 0
        last_cost_line_idx = -1
        cost_lines_before = []
        if os.path.exists(log_path):
            log_start_pos = os.path.getsize(log_path)
            # Find last cost_usd line before search
            with open(log_path, "r") as f:
                all_log_lines = f.readlines()
                cost_lines_before = [
                    i for i, line in enumerate(all_log_lines) if "cost_usd" in line
                ]
                last_cost_line_idx = cost_lines_before[-1] if cost_lines_before else -1
        start_time = datetime.datetime.now()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                get_answer,
                query=question,
                top_k=top_k,
                llm_model=llm_model,
            )
            log_pos = log_start_pos
            with st.spinner("מריץ LLM ומחזיר תשובה + מקורות..."):
                while not future.done():
                    if os.path.exists(log_path):
                        with open(log_path, "r") as f:
                            f.seek(log_pos)
                            new_lines = f.readlines()
                            if new_lines:
                                log_lines.extend(new_lines)
                            log_pos = f.tell()
                        # Show last 30 lines, LTR, full width
                        log_text = "".join(log_lines[-30:])
                        log_area.markdown(
                            f'<div style="direction: ltr; text-align: left; white-space: pre; font-family: monospace; width: 100%;">{log_text}</div>',
                            unsafe_allow_html=True,
                        )
                    time.sleep(0.3)
                result = future.result()
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
        # After result is ready, show only new log lines from this run
        cost_lines_after = []
        cost_entries = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                all_log_lines = f.readlines()
                # Find all new cost_usd lines after last_cost_line_idx
                cost_lines_after = [
                    (i, line)
                    for i, line in enumerate(all_log_lines)
                    if "cost_usd" in line and i > last_cost_line_idx
                ]
                # Parse cost lines
                for _, line in cost_lines_after:
                    # Example: OpenAI embedding | model=text-embedding-3-large | ... | cost_usd=0.00000130
                    # or: OpenAI LLM | model=gpt-4o | ... | cost_usd=0.00012345
                    if "OpenAI embedding" in line:
                        cat = "embedding"
                    elif "OpenAI LLM" in line:
                        cat = "llm"
                    else:
                        cat = "other"
                    # Extract model
                    import re

                    m = re.search(r"model=([^ |]+)", line)
                    model = m.group(1) if m else "unknown"
                    # Extract cost
                    m = re.search(r"cost_usd=([0-9.]+)", line)
                    cost = float(m.group(1)) if m else 0.0
                    cost_entries.append((cat, model, cost))
            # Aggregate costs
            from collections import defaultdict

            cost_by_cat_model = defaultdict(float)
            total_cost = 0.0
            for cat, model, cost in cost_entries:
                cost_by_cat_model[(cat, model)] += cost
                total_cost += cost
            # Format cost summary
            cost_parts = [
                f"{cat}:{model}=$" + f"{cost_by_cat_model[(cat, model)]:.3f}"
                for (cat, model) in sorted(cost_by_cat_model)
            ]
            cost_summary = " | ".join(cost_parts)
            cost_summary += f" | total=$" + f"{total_cost:.3f}"
        else:
            cost_summary = ""
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                f.seek(log_start_pos)
                log_text = f.read()
            log_area.markdown(
                f'<div style="direction: ltr; text-align: left; white-space: pre; font-family: monospace; width: 100%;">{log_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            log_area.info("Log file not found.")
        answer, sources, fragments = result
        # Detect web search answer: fragments is empty DataFrame and sources is non-empty
        is_web_search = (
            (
                fragments is not None
                and isinstance(fragments, pd.DataFrame)
                and fragments.empty
            )
            and sources
            and isinstance(sources, str)
            and sources.strip()
        )
        if is_web_search:
            web_warning = '<div style="color:#b30000; font-size:1.2em; font-weight:bold; padding:8px 0;">⚠️ תשובה זו מבוססת על חיפוש אינטרנטי ⚠</div>'
        else:
            web_warning = ""
        answer_container.markdown(
            f"<h2>תשובה</h2>"
            f'<div style="color:gray; font-size: 1em;">{minutes}:{seconds:02d} דקות | {cost_summary}</div>'
            f"{web_warning}"
            f"<div>{answer}</div>",
            unsafe_allow_html=True,
        )

        # Show sources/fragments/context
        if (
            fragments is not None
            and isinstance(fragments, pd.DataFrame)
            and not fragments.empty
        ):
            df_sources = fragments
            n_sources = len(df_sources)
            with st.expander(f"הצג מקורות ({n_sources} Sources)"):
                show_cols = [
                    c
                    for c in [
                        "score",
                        "llm_relevance_score",
                        "title",
                        "page_number",
                        "sequence_number",
                        "text",
                    ]
                    if c in df_sources.columns
                ]
                df_show = (
                    df_sources[show_cols].copy() if show_cols else df_sources.copy()
                )
                if "text" in df_show:
                    df_show["text"] = df_show["text"].apply(
                        lambda t: t[:120] + ("..." if len(str(t)) > 120 else "")
                    )
                st.dataframe(df_show, use_container_width=True)
        elif sources:
            MAX_TEXT_LENGTH = 3000
            text = sources[:MAX_TEXT_LENGTH] + (
                "\n...\n"
                if isinstance(sources, str) and len(sources) > MAX_TEXT_LENGTH
                else ""
            )
            st.markdown(
                f'<div style="direction: rtl; text-align: right; white-space: pre-wrap;">{text}</div>',
                unsafe_allow_html=True,
            )
    # --- END RAG QA UI ---
elif st.session_state["auth_status"] is False:
    st.error("Invalid username or password")

import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "Chinook.db"

st.set_page_config(
    page_title="QueryMind | Text-to-SQL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- STYLING ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0D1117;
    color: #F3F4F6;
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

section[data-testid="stSidebar"] {
    background-color: #11161D;
    border-right: 1px solid #252B36;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #303846;
    background-color: #171D26;
    color: #F3F4F6;
}

.stButton > button:hover {
    border-color: #818CF8;
    color: white;
}

[data-testid="stChatMessage"] {
    border: 1px solid #252B36;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.stCaption {
    color: #9CA3AF;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE ----------------

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.info("Place Chinook.db in the same folder as app.py.")
    st.stop()

db = SQLDatabase.from_uri(
    f"sqlite:///{DB_PATH.as_posix()}",
    sample_rows_in_table_info=0,
)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Text-to-SQL assistant.
Generate one valid SQLite SELECT query using only the
provided database schema.

Rules:
- Use exact table and column names.
- Return only SQL, without Markdown fences.
- Do not invent tables or columns.
- Generate read-only queries only.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER,
  CREATE, ATTACH, or other database-changing commands.
"""
        ),
        (
            "human",
            """Database schema:
{schema}

Question: {question}

SQL query:"""
        ),
    ])

    return prompt | model | StrOutputParser()


sql_chain = load_model()


def generate_sql(question):
    raw_sql = sql_chain.invoke({
        "schema": db.get_table_info(),
        "question": question,
    }).strip()

    raw_sql = re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        raw_sql,
        flags=re.IGNORECASE,
    ).strip()

    return raw_sql


# ---------------- SAFE QUERY EXECUTION ----------------

def execute_sql(query):
    """
    Execute a read-only SQL query against the database.
    A SQLite authorizer provides an additional layer of
    protection against database modifications.
    """

    query = query.strip()

    if not query:
        raise ValueError("The model did not generate a SQL query.")

    # Accept SELECT statements and read-only CTEs.
    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in query.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")

    connection = sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro",
        uri=True,
    )

    # Block database-changing operations and unsafe features.
    blocked_actions = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    }

    def authorize(action, arg1, arg2, database, source):
        if action in blocked_actions:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        connection.set_authorizer(authorize)

        cursor = connection.execute(query)

        if cursor.description is None:
            raise ValueError("The query did not return a result set.")

        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchmany(1000)

        return pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = {}


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🧠 QueryMind")
    st.caption("Ask questions. Get SQL. Explore data.")

    if st.button("＋ New Chat", use_container_width=True):
        if st.session_state.messages:
            title = next(
                (
                    message["content"][:35]
                    for message in st.session_state.messages
                    if message["role"] == "user"
                ),
                "Untitled chat",
            )

            st.session_state.history[title] = (
                st.session_state.messages.copy()
            )

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### Database")
    st.caption("📁 Chinook.db")
    st.caption(f"📊 {len(db.get_usable_table_names())} tables available")

    st.markdown("### AI Model")
    st.caption("🦙 Ollama")
    st.caption("Model: qwen2.5:3b")

    st.divider()

    st.markdown("### Recent Chats")

    if not st.session_state.history:
        st.caption("Your previous chats will appear here.")
    else:
        for title, messages in reversed(
            list(st.session_state.history.items())
        ):
            if st.button(title, key=f"history_{title}",
                         use_container_width=True):
                st.session_state.messages = messages.copy()
                st.rerun()

    if st.button("Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- MAIN CHAT ----------------

st.markdown("# QueryMind")
st.caption("Your local AI assistant for querying databases with natural language.")

if not st.session_state.messages:
    st.markdown("### What would you like to know?")
    st.write("Ask a question about the data stored in your database.")

    examples = [
        "Show me 10 random artists",
        "How many tracks are in the database?",
        "Which artists have the most albums?",
        "Show the 5 most expensive tracks",
    ]

    cols = st.columns(2)

    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sql"):
            with st.expander("View generated SQL"):
                st.code(message["sql"], language="sql")

        if message.get("results") is not None:
            st.dataframe(
                pd.DataFrame(message["results"]),
                use_container_width=True,
                hide_index=True,
            )


question = st.chat_input("Ask a question about your database...")

if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating SQL and querying your database..."):
                sql = generate_sql(question)
                results = execute_sql(sql)

            st.success("Query executed successfully.")

            with st.expander("View generated SQL"):
                st.code(sql, language="sql")

            if results.empty:
                st.info("The query returned no results.")
            else:
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here are your query results.",
                "sql": sql,
                "results": results.to_dict(orient="records"),
            })

        except Exception as error:
            error_message = str(error)

            st.error("I couldn't complete that query.")
            st.caption(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I couldn't complete that query: {error_message}",
            })
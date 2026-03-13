"""
Streamlit frontend for the Inventory NLQ Agent.
Provides an interactive chat interface to query inventory data using natural language.
Compatible with Streamlit 1.12+.
"""

import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from nlq_agent import create_agent, run_query, DB_PATH, SCHEMA_DESCRIPTION

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chemlex Inventory NLQ Agent",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0;
        padding-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0.3rem 0 0 0;
        font-size: 1.5rem;
    }
    .user-msg {
        background: #EEF2FF;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .bot-msg {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .msg-label {
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .user-label { color: #4F46E5; }
    .bot-label { color: #16A34A; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache(ttl=60, allow_output_mutation=True)
def get_db_stats():
    """Get database statistics for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    stats = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    stats["tables"] = len(tables)
    total_rows = 0
    for t in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{t}"')
        total_rows += cursor.fetchone()[0]
    stats["total_rows"] = total_rows

    cursor.execute("SELECT COUNT(DISTINCT Material) FROM material_stock_requirement")
    stats["materials"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT Plant) FROM material_stock_requirement")
    stats["plants"] = cursor.fetchone()[0]

    conn.close()
    return stats


def extract_sql_from_messages(messages):
    """Extract SQL queries from agent messages for display."""
    sql_queries = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "execute_sql":
                    sql_queries.append(tc["args"].get("query", ""))
    return sql_queries


def try_create_dataframe(sql_query: str):
    """Try to execute the SQL and return a DataFrame."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        if df.empty or len(df.columns) < 2:
            return None
        return df
    except Exception:
        return None


def auto_chart(df: pd.DataFrame):
    """Automatically create an appropriate chart based on the data shape."""
    if df is None or df.empty:
        return None

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]

    if len(num_cols) >= 1 and len(cat_cols) >= 1:
        if len(df) <= 20:
            fig = px.bar(
                df, x=cat_cols[0], y=num_cols[0],
                color=cat_cols[1] if len(cat_cols) > 1 else None,
                title=f"{num_cols[0]} by {cat_cols[0]}",
                template="plotly_white",
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            return fig
        else:
            fig = px.line(
                df, x=cat_cols[0], y=num_cols[0],
                color=cat_cols[1] if len(cat_cols) > 1 else None,
                title=f"{num_cols[0]} over {cat_cols[0]}",
                template="plotly_white",
            )
            fig.update_layout(height=450)
            return fig
    elif len(num_cols) >= 2:
        fig = px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            title=f"{num_cols[1]} vs {num_cols[0]}",
            template="plotly_white",
        )
        fig.update_layout(height=450)
        return fig
    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")

    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Database Overview")

    if os.path.exists(DB_PATH):
        stats = get_db_stats()
        col1, col2 = st.columns(2)
        col1.metric("Tables", stats["tables"])
        col2.metric("Total Rows", f"{stats['total_rows']:,}")
        col1.metric("Materials", f"{stats['materials']:,}")
        col2.metric("Plants", stats["plants"])

        st.markdown("---")
        st.markdown("### Tables")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for t in cursor.fetchall():
            tname = t[0]
            cursor.execute(f'SELECT COUNT(*) FROM "{tname}"')
            count = cursor.fetchone()[0]
            with st.expander(f"{tname} ({count:,} rows)"):
                cursor.execute(f'PRAGMA table_info("{tname}")')
                cols = cursor.fetchall()
                for c in cols:
                    st.text(f"  {c[1]} ({c[2]})")
        conn.close()
    else:
        st.warning("Database not found.")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.experimental_rerun()


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">Chemlex Inventory NLQ Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about your inventory data in plain English</p>',
    unsafe_allow_html=True,
)

# Database metrics row
if os.path.exists(DB_PATH):
    stats = get_db_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f'<div class="metric-card"><h3>Tables</h3><h2>{stats["tables"]}</h2></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><h3>Total Records</h3><h2>{stats["total_rows"]:,}</h2></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><h3>Materials</h3><h2>{stats["materials"]:,}</h2></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card"><h3>Plants</h3><h2>{stats["plants"]}</h2></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Sample Questions ─────────────────────────────────────────────────────────

sample_questions = [
    "How many unique materials are in each plant?",
    "What is the total WIP value by plant?",
    "Show me the top 10 materials by safety stock",
    "Which materials have the highest customer order quantities?",
    "What is the total ATP quantity by plant?",
    "Who is responsible for the most materials?",
    "Which work centers handle the most production orders?",
    "Compare inflow vs outflow quantities by plant",
    "What materials have WIP value greater than 100000?",
    "Show monthly trends in planned independent requirements",
]

st.markdown("**Quick Questions:**")
qcols = st.columns(5)
selected_sample = None
for i, q in enumerate(sample_questions):
    col_idx = i % 5
    if qcols[col_idx].button(q, key=f"sq_{i}"):
        selected_sample = q

st.markdown("---")

# ── Chat Interface ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history
for entry in st.session_state.messages:
    role = entry["role"]
    content = entry["content"]
    if role == "user":
        st.markdown(
            f'<div class="user-msg"><div class="msg-label user-label">You</div>{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-msg"><div class="msg-label bot-label">Agent</div>{content}</div>',
            unsafe_allow_html=True,
        )
        if "sql" in entry and entry["sql"]:
            with st.expander("View SQL Query"):
                st.code(entry["sql"], language="sql")
        if "dataframe" in entry and entry["dataframe"] is not None:
            with st.expander("View Data Table"):
                st.dataframe(entry["dataframe"])
        if "chart" in entry and entry["chart"] is not None:
            st.plotly_chart(entry["chart"])

# Question input
with st.form("question_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask a question about your inventory data",
    )
    submitted = st.form_submit_button("Ask")

question = selected_sample or (user_input if submitted else None)

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f'<div class="user-msg"><div class="msg-label user-label">You</div>{question}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Thinking... analyzing your question and querying the database..."):
        try:
            agent = create_agent(model_choice)
            result_messages = run_query(agent, question, st.session_state.chat_history)

            # Extract the final answer
            final_answer = ""
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    final_answer = msg.content
                    break

            if not final_answer:
                final_answer = "I wasn't able to generate a response. Please try rephrasing your question."

            # Extract SQL queries
            sql_queries = extract_sql_from_messages(result_messages)
            last_sql = sql_queries[-1] if sql_queries else None

            # Display answer
            st.markdown(
                f'<div class="bot-msg"><div class="msg-label bot-label">Agent</div>{final_answer}</div>',
                unsafe_allow_html=True,
            )

            # Show SQL
            if last_sql:
                with st.expander("View SQL Query"):
                    st.code(last_sql, language="sql")

            # Try to create chart
            df = None
            chart = None
            if last_sql:
                df = try_create_dataframe(last_sql)
                if df is not None:
                    with st.expander("View Data Table"):
                        st.dataframe(df)
                    chart = auto_chart(df)
                    if chart:
                        st.plotly_chart(chart)

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sql": last_sql,
                "dataframe": df,
                "chart": chart,
            })

            # Update chat history for context
            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.chat_history.append(AIMessage(content=final_answer))

            # Keep chat history manageable
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history = st.session_state.chat_history[-20:]

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ── Direct SQL Explorer ──────────────────────────────────────────────────────

st.markdown("---")

with st.expander("Direct SQL Explorer"):
    st.markdown("Run custom SQL queries directly against the database.")
    custom_sql = st.text_area(
        "SQL Query",
        value="SELECT Plant, COUNT(DISTINCT Material) as material_count\nFROM material_stock_requirement\nGROUP BY Plant\nORDER BY material_count DESC",
        height=120,
    )
    if st.button("Run Query"):
        try:
            conn = sqlite3.connect(DB_PATH)
            result_df = pd.read_sql_query(custom_sql, conn)
            conn.close()
            st.dataframe(result_df)

            chart = auto_chart(result_df)
            if chart:
                st.plotly_chart(chart)
        except Exception as e:
            st.error(f"Query error: {e}")

with st.expander("Schema Reference"):
    st.markdown(SCHEMA_DESCRIPTION)

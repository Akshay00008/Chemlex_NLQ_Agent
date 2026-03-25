"""
Streamlit frontend for the Inventory NLQ Agent.
Provides an interactive chat interface to query inventory data using natural language.
"""

import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.express as px

from langchain_core.messages import HumanMessage, AIMessage
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

@st.cache_data(ttl=60)
def get_db_stats():
    """Get database statistics for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    stats = {}

    cursor.execute("SELECT COUNT(*) FROM current_inventory")
    stats["total_rows"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT Material) FROM current_inventory")
    stats["materials"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT Plant) FROM current_inventory")
    stats["plants"] = cursor.fetchone()[0]

    cursor.execute("SELECT ROUND(SUM(Shelf_Stock_USD), 0) FROM current_inventory")
    stats["total_shelf_usd"] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(DISTINCT Material_Type) FROM current_inventory WHERE Material_Type IS NOT NULL")
    stats["material_types"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT Product_Family) FROM current_inventory WHERE Product_Family IS NOT NULL")
    stats["product_families"] = cursor.fetchone()[0]

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

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

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
        col1.metric("Total Rows", f"{stats['total_rows']:,}")
        col2.metric("Materials", f"{stats['materials']:,}")
        col1.metric("Plants", stats["plants"])
        col2.metric("Material Types", stats["material_types"])

        st.markdown("---")
        st.markdown("### Table Schema")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('PRAGMA table_info("current_inventory")')
        cols = cursor.fetchall()
        with st.expander(f"current_inventory ({stats['total_rows']:,} rows)"):
            for c in cols:
                st.text(f"  {c[1]} ({c[2]})")
        conn.close()
    else:
        st.warning("Database not found.")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.rerun()


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">Chemlex Inventory NLQ Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about your inventory data in plain English</p>',
    unsafe_allow_html=True,
)

# Database metrics row
if os.path.exists(DB_PATH):
    stats = get_db_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(
        f'<div class="metric-card"><h3>Total Records</h3><h2>{stats["total_rows"]:,}</h2></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><h3>Materials</h3><h2>{stats["materials"]:,}</h2></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><h3>Plants</h3><h2>{stats["plants"]}</h2></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card"><h3>Shelf Stock Value</h3><h2>${stats["total_shelf_usd"]:,.0f}</h2></div>',
        unsafe_allow_html=True,
    )
    m5.markdown(
        f'<div class="metric-card"><h3>Product Families</h3><h2>{stats["product_families"]}</h2></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Sample Questions ─────────────────────────────────────────────────────────

sample_questions = [
    "What is the total shelf stock value by plant?",
    "Show me top 10 materials by shelf stock dollar value",
    "What is the ABC classification breakdown?",
    "Which plants have the highest WIP value?",
    "Show materials with DOH greater than 90 days",
    "What is the total GIT value by product family?",
    "Who are the top purchasing groups by inventory value?",
    "Compare shelf stock vs safety stock by plant",
    "Which material types have the most inventory?",
    "Show demand vs shelf stock for top 10 materials",
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
            st.plotly_chart(entry["chart"], use_container_width=True)

# Question input
with st.form("question_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question about your inventory data")
    submitted = st.form_submit_button("Ask")

question = selected_sample or (user_input if submitted else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f'<div class="user-msg"><div class="msg-label user-label">You</div>{question}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Analyzing your question and querying the database..."):
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

            sql_queries = extract_sql_from_messages(result_messages)
            last_sql = sql_queries[-1] if sql_queries else None

            st.markdown(
                f'<div class="bot-msg"><div class="msg-label bot-label">Agent</div>{final_answer}</div>',
                unsafe_allow_html=True,
            )

            if last_sql:
                with st.expander("View SQL Query"):
                    st.code(last_sql, language="sql")

            df = None
            chart = None
            if last_sql:
                df = try_create_dataframe(last_sql)
                if df is not None:
                    with st.expander("View Data Table"):
                        st.dataframe(df)
                    chart = auto_chart(df)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sql": last_sql,
                "dataframe": df,
                "chart": chart,
            })

            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.chat_history.append(AIMessage(content=final_answer))

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
        value="SELECT Plant, COUNT(DISTINCT Material) as material_count,\n       ROUND(SUM(Shelf_Stock_USD), 2) as total_shelf_value\nFROM current_inventory\nGROUP BY Plant\nORDER BY total_shelf_value DESC\nLIMIT 15",
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
                st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Query error: {e}")

with st.expander("Schema Reference"):
    st.markdown(SCHEMA_DESCRIPTION)

"""
NLQ Agent for Current Inventory data using LangChain + LangGraph.
Converts natural language questions into SQL queries against the inventory SQLite database.
"""

import sqlite3
import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_PATH = os.path.join(os.path.dirname(__file__), "inventory_data.db")

# ── Schema description for the LLM ──────────────────────────────────────────

SCHEMA_DESCRIPTION = """
You have access to a Current Inventory SQLite database with one table:

**current_inventory** (126,469 rows) — Current inventory snapshot across all plants

Columns:
- Plant (TEXT) — Manufacturing plant code (e.g., '2001', '3001', '7200')
- Material (TEXT) — Material number / SKU identifier
- Material_Name (TEXT) — Human-readable material description
- Material_Type (TEXT) — Category: 'Finished products', 'Raw materials', 'Semifinished products', 'Trading goods', 'Packaging', 'Spare parts', 'Operating supplies-NON VA', 'Optng suppl/Non Cos-VALUA', 'Prod. resources/tools', 'Nonvaluated materials', 'Services'
- UOM (TEXT) — Unit of measure (EA, KG, LB, FT, etc.)
- Shelf_Stock (REAL) — Quantity of stock on shelf (in UOM)
- Shelf_Stock_USD (REAL) — Dollar value of shelf stock
- GIT (REAL) — Goods In Transit quantity
- GIT_USD (REAL) — Dollar value of goods in transit
- WIP (REAL) — Work In Progress quantity
- WIPUSD (REAL) — Dollar value of WIP
- DOH (REAL) — Days on Hand (how many days current stock will last)
- Safety_Stock (REAL) — Minimum inventory level to prevent stockouts
- Demand (REAL) — Current demand quantity
- Product_Family (TEXT) — Product family grouping
- SOP_Family (TEXT) — Sales & Operations Planning family
- Product_Group (TEXT) — Product group classification
- Material_Group (TEXT) — Material group classification
- Product_Category (TEXT) — Product category
- Material_Application (TEXT) — Application area for the material
- Sub_Application (TEXT) — Sub-application area
- ABC (TEXT) — ABC classification ('A' = high value, 'B' = medium, 'C' = low)
- MRP_Controller_Text (TEXT) — MRP controller / planner name
- Purchasing_Group_Text (TEXT) — Purchasing group / buyer name

There are 46 distinct plants and 80,914 distinct materials.

Domain context:
- This is a manufacturing/chemical company inventory system
- Shelf Stock = physical inventory on hand
- GIT = Goods In Transit (ordered but not yet received)
- WIP = Work In Progress (being manufactured)
- DOH = Days on Hand (stock / daily demand)
- Safety Stock = minimum buffer inventory
- ABC classification: A = high-value items needing tight control, B = moderate, C = low-value
- MRP = Material Requirements Planning
- SOP = Sales & Operations Planning
"""

SYSTEM_PROMPT = f"""You are an expert inventory data analyst. You answer questions about inventory,
materials, stock levels, and supply chain data by writing and executing SQL queries.

{SCHEMA_DESCRIPTION}

Rules:
1. Always write valid SQLite SQL. Use double quotes for column names if needed.
2. LIMIT results to 50 rows unless the user asks for more or you need aggregation.
3. For aggregations (COUNT, SUM, AVG), don't add unnecessary LIMIT.
4. When you get results, provide a clear, concise answer with key insights.
5. If the query returns no results, explain why and suggest alternatives.
6. For chart-worthy data, format your answer so it's clear what to visualize.
7. If you're unsure about column names, use the describe_tables tool first.
8. Always be helpful and explain what the data means in business context.
9. Use ROUND() for dollar values and quantities to keep output readable.
10. When filtering by text fields, use LIKE with % for partial matches since values may vary.
"""


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the inventory database and return results.
    Use this for SELECT queries to answer data questions.
    Returns results as a formatted string with column headers."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query returned no results."

        result = f"Columns: {', '.join(columns)}\n"
        result += f"Row count: {len(rows)}\n\n"
        for row in rows[:100]:
            result += " | ".join(str(v) for v in row) + "\n"
        if len(rows) > 100:
            result += f"\n... and {len(rows) - 100} more rows"
        return result
    except Exception as e:
        return f"SQL Error: {str(e)}"


@tool
def describe_tables(table_name: str = "") -> str:
    """Get schema information about database tables.
    Pass a table name to get detailed column info, or empty string for all tables."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if table_name:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            cols = cursor.fetchall()
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            result = f"Table: {table_name} ({count} rows)\nColumns:\n"
            for c in cols:
                result += f"  - {c[1]} ({c[2]})\n"

            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
            samples = cursor.fetchall()
            col_names = [c[1] for c in cols]
            result += "\nSample rows:\n"
            for s in samples:
                result += "  " + str(dict(zip(col_names, s))) + "\n"
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            result = "Available tables:\n"
            for t in tables:
                cursor.execute(f'SELECT COUNT(*) FROM "{t[0]}"')
                count = cursor.fetchone()[0]
                result += f"  - {t[0]} ({count} rows)\n"

        conn.close()
        return result
    except Exception as e:
        return f"Error: {str(e)}"


tools = [execute_sql, describe_tables]


# ── LangGraph Agent ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def _get_secret(key, default=None):
    """Get secret from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


def create_agent(model_name: str = None):
    """Create the LangGraph NLQ agent."""
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in .env file or Streamlit secrets")
    if model_name is None:
        model_name = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_query(agent, question: str, chat_history: list = None):
    """Run a natural language question through the agent."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})
    return result["messages"]

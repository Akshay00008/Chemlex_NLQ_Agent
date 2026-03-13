"""
NLQ Agent for Inventory Dashboard data using LangChain + LangGraph.
Converts natural language questions into SQL queries against the inventory SQLite database.
"""

import sqlite3
import os
import json
import re
from typing import TypedDict, Annotated, Sequence
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
You have access to an Inventory Dashboard SQLite database with the following tables:

1. **material_stock_requirement** (297,009 rows) — MRP stock requirement data (MCBZ report)
   Columns: InfoProvider, Material, Plant, ATP_Quantity_in_SKUs, Cumulated_ATP_Quantity,
   Future_Qty__of_Goods_Issues_that_is_Valid_for_MRP, Future_Quantity_of_Goods_Receipts_that_is_Valid_for_MRP,
   Future_Stock_MRP, Planned_Independent_Requirements, Safety_Stock, Total_Future_Goods_Receipts,
   Total_Quantity_of_Future_Goods_Issues, Base_Unit_of_Measure, Calendar_day, Calendar_year___week,
   Calendar_year_month, Required_quantity_for_materials_management_in_SKUs, PlantMaterailKey, calendar_year_month_7d

2. **material_wip** (902 rows) — Work-in-Progress materials
   Columns: BR_Date, Material, Material_Description, Order_Number, Plant, Posting_Date, Quantity_Due, WIP_in_LC_at_BR

3. **md04_mdez** (210,444 rows) — MRP element detail (MD04 report)
   Columns: Calendar_month, Calendar_year___week, Calendar_year_month, InfoProvider, Material, Plant, Plant_Text,
   ATP_Quantity_in_SKUs, CUM_ATP_QTY, Customer_ORDER_QTY, Purchase_or_Planned_ORDER_QTY,
   QTY_For_Specific_MRP_ELEMENT, Quantity_received_or_quantity_required, Required_Inflow_Quantity,
   Required_Outflow_Quantity, Calendar_day, PlantMaterailKey

4. **production_orders_3** (12,321 rows) — Production order work center assignments
   Columns: Base_Unit_of_Measure, Material, Plant, Work_Center_Resource, Work_Center_Resource_Description

5. **zccav01** (6,757 rows) — Material responsibility assignments
   Columns: Material, Person, Person_Responsible, Plant, Plant_description, mat_plant

Key relationships:
- Material and Plant are common join keys across all tables
- PlantMaterailKey (PlantMaterialKey) links material_stock_requirement, md04_mdez tables
- Calendar_year_month and Calendar_day are date fields in material_stock_requirement and md04_mdez

Domain context:
- This is a manufacturing/chemical company inventory system
- ATP = Available To Promise (inventory available for customer orders)
- MRP = Material Requirements Planning
- WIP = Work In Progress
- SKU = Stock Keeping Unit
- DOH = Days on Hand
- LC at BR = Local Currency at Business Rate
- Safety Stock = minimum inventory level to prevent stockouts
- Plants are manufacturing facilities (e.g., Plant 2001, 2024)
"""

SYSTEM_PROMPT = f"""You are an expert inventory data analyst. You answer questions about inventory,
materials, production, and stock data by writing and executing SQL queries.

{SCHEMA_DESCRIPTION}

Rules:
1. Always write valid SQLite SQL. Use double quotes for column names with special characters.
2. LIMIT results to 50 rows unless the user asks for more or you need aggregation.
3. For aggregations (COUNT, SUM, AVG), don't add unnecessary LIMIT.
4. When you get results, provide a clear, concise answer with key insights.
5. If the query returns no results, explain why and suggest alternatives.
6. For chart-worthy data, format your answer so it's clear what to visualize.
7. If you're unsure about column names, use the describe_tables tool first.
8. Always be helpful and explain what the data means in business context.
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

        # Format as table
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

            # Sample data
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

import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def create_agent(model_name: str = None):
    """Create the LangGraph NLQ agent using API key from .env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in .env file")
    if model_name is None:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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

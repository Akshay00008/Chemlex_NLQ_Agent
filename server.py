"""
Chemelex Inventory Intelligence — Flask Backend
Provides authentication, KPI endpoints, NLQ chat, and MongoDB-backed conversation persistence.
"""

import os
import sqlite3
import secrets
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from nlq_agent import create_agent, run_query, DB_PATH

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# ── MongoDB ──────────────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo = MongoClient(MONGO_URI)
db = mongo["chemlex"]
users_col = db["users"]
conversations_col = db["conversations"]

# Ensure indexes
users_col.create_index("username", unique=True)
conversations_col.create_index("user_id")
conversations_col.create_index("updated_at")

# ── Seed default admin user ──────────────────────────────────────────────────

if users_col.count_documents({"username": "admin"}) == 0:
    users_col.insert_one({
        "username": "admin",
        "password_hash": generate_password_hash("chemlex2024"),
        "full_name": "Administrator",
        "role": "admin",
        "created_at": datetime.utcnow(),
    })

# ── Auth Decorator ───────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ── Helper: JSON-safe ObjectId ───────────────────────────────────────────────

def serialize_doc(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ── KPI Helper ───────────────────────────────────────────────────────────────

def compute_kpis():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    kpis = {}

    # Shelf Stock Value
    c.execute("SELECT ROUND(SUM(COALESCE(Shelf_Stock_USD, 0)), 2) FROM current_inventory")
    kpis["shelf_stock_usd"] = c.fetchone()[0] or 0

    # GIT Value
    c.execute("SELECT ROUND(SUM(COALESCE(GIT_USD, 0)), 2) FROM current_inventory")
    kpis["git_usd"] = c.fetchone()[0] or 0

    # WIP Value
    c.execute("SELECT ROUND(SUM(COALESCE(WIPUSD, 0)), 2) FROM current_inventory")
    kpis["wip_usd"] = c.fetchone()[0] or 0

    # Total Inventory Value (Shelf + GIT + WIP) — sum the individual parts
    kpis["total_inventory_usd"] = round(kpis["shelf_stock_usd"] + kpis["git_usd"] + kpis["wip_usd"], 2)

    # Active Plants (plants with any stock > 0)
    c.execute("""
        SELECT COUNT(DISTINCT Plant) FROM current_inventory
        WHERE Shelf_Stock > 0 OR COALESCE(GIT, 0) > 0 OR COALESCE(WIP, 0) > 0
    """)
    kpis["active_plants"] = c.fetchone()[0]

    # Total Plants
    c.execute("SELECT COUNT(DISTINCT Plant) FROM current_inventory")
    kpis["total_plants"] = c.fetchone()[0]

    # Plant-Material combinations with stock (each row = one plant+material, so count rows)
    c.execute("SELECT COUNT(*) FROM current_inventory WHERE Shelf_Stock > 0")
    kpis["plant_materials_with_stock"] = c.fetchone()[0]

    # Total Plant-Material combinations
    c.execute("SELECT COUNT(*) FROM current_inventory")
    kpis["total_plant_materials"] = c.fetchone()[0]

    # Unique materials with stock in at least one plant
    c.execute("SELECT COUNT(DISTINCT Material) FROM current_inventory WHERE Shelf_Stock > 0")
    kpis["unique_materials_with_stock"] = c.fetchone()[0]

    # Average DOH (only for items with DOH > 0)
    c.execute("SELECT ROUND(AVG(DOH), 1) FROM current_inventory WHERE DOH > 0")
    kpis["avg_doh"] = c.fetchone()[0] or 0

    # Items below safety stock (plant+material combinations)
    c.execute("""
        SELECT COUNT(*) FROM current_inventory
        WHERE Safety_Stock > 0 AND Shelf_Stock < Safety_Stock
    """)
    kpis["below_safety_stock"] = c.fetchone()[0]

    # Plants affected by below-safety-stock items
    c.execute("""
        SELECT COUNT(DISTINCT Plant) FROM current_inventory
        WHERE Safety_Stock > 0 AND Shelf_Stock < Safety_Stock
    """)
    kpis["plants_with_shortages"] = c.fetchone()[0]

    # Below safety stock breakdown by plant (top 8)
    c.execute("""
        SELECT Plant, COUNT(*) AS cnt
        FROM current_inventory
        WHERE Safety_Stock > 0 AND Shelf_Stock < Safety_Stock
        GROUP BY Plant ORDER BY cnt DESC LIMIT 8
    """)
    kpis["below_ss_by_plant"] = [{"plant": r[0], "count": r[1]} for r in c.fetchall()]

    # Top 5 plants by inventory value
    c.execute("""
        SELECT Plant,
               ROUND(SUM(COALESCE(Shelf_Stock_USD, 0) + COALESCE(GIT_USD, 0) + COALESCE(WIPUSD, 0)), 2) AS total
        FROM current_inventory
        GROUP BY Plant ORDER BY total DESC LIMIT 5
    """)
    kpis["top_plants"] = [{"plant": r[0], "value": r[1]} for r in c.fetchall()]

    # Inventory by Material Type (top 6)
    c.execute("""
        SELECT Material_Type,
               ROUND(SUM(COALESCE(Shelf_Stock_USD, 0) + COALESCE(GIT_USD, 0) + COALESCE(WIPUSD, 0)), 2) AS total
        FROM current_inventory
        WHERE Material_Type IS NOT NULL
        GROUP BY Material_Type ORDER BY total DESC LIMIT 6
    """)
    kpis["by_material_type"] = [{"type": r[0], "value": r[1]} for r in c.fetchall()]

    conn.close()
    return kpis


# ── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))


@app.route("/login")
def login_page():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html",
                           user_name=session.get("full_name", "User"))


# ── Auth API ─────────────────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    user = users_col.find_one({"username": username})
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = str(user["_id"])
    session["username"] = user["username"]
    session["full_name"] = user.get("full_name", user["username"])
    session["role"] = user.get("role", "user")

    return jsonify({"success": True, "name": session["full_name"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


# ── KPI API ──────────────────────────────────────────────────────────────────

@app.route("/api/kpis")
@login_required
def api_kpis():
    return jsonify(compute_kpis())


# ── Conversation API ─────────────────────────────────────────────────────────

@app.route("/api/conversations", methods=["GET"])
@login_required
def list_conversations():
    convos = conversations_col.find(
        {"user_id": session["user_id"]},
        {"messages": 0}  # exclude messages for list view
    ).sort("updated_at", -1).limit(50)

    result = []
    for c in convos:
        result.append({
            "_id": str(c["_id"]),
            "title": c.get("title", "New conversation"),
            "message_count": c.get("message_count", 0),
            "updated_at": c.get("updated_at", c.get("created_at", "")).isoformat() if isinstance(c.get("updated_at"), datetime) else str(c.get("updated_at", "")),
        })
    return jsonify(result)


@app.route("/api/conversations", methods=["POST"])
@login_required
def create_conversation():
    now = datetime.utcnow()
    doc = {
        "user_id": session["user_id"],
        "title": "New conversation",
        "messages": [],
        "message_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    result = conversations_col.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return jsonify(doc), 201


@app.route("/api/conversations/<convo_id>", methods=["GET"])
@login_required
def get_conversation(convo_id):
    try:
        convo = conversations_col.find_one({
            "_id": ObjectId(convo_id),
            "user_id": session["user_id"],
        })
    except Exception:
        return jsonify({"error": "Invalid ID"}), 400

    if not convo:
        return jsonify({"error": "Not found"}), 404

    convo["_id"] = str(convo["_id"])
    # Convert datetime objects in messages
    for msg in convo.get("messages", []):
        if isinstance(msg.get("timestamp"), datetime):
            msg["timestamp"] = msg["timestamp"].isoformat()

    return jsonify(convo)


@app.route("/api/conversations/<convo_id>", methods=["DELETE"])
@login_required
def delete_conversation(convo_id):
    try:
        conversations_col.delete_one({
            "_id": ObjectId(convo_id),
            "user_id": session["user_id"],
        })
    except Exception:
        return jsonify({"error": "Invalid ID"}), 400
    return jsonify({"success": True})


# ── NLQ Query API ────────────────────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
@login_required
def api_query():
    data = request.get_json()
    question = data.get("question", "").strip()
    convo_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Load chat history from conversation
    chat_history = []
    if convo_id:
        try:
            convo = conversations_col.find_one({
                "_id": ObjectId(convo_id),
                "user_id": session["user_id"],
            })
            if convo:
                for msg in convo.get("messages", [])[-20:]:  # Last 20 messages for context
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
        except Exception:
            pass

    try:
        agent = create_agent()
        result_messages = run_query(agent, question, chat_history if chat_history else None)

        # Extract final answer
        final_answer = ""
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                final_answer = msg.content
                break

        if not final_answer:
            final_answer = "I wasn't able to generate a response. Please try rephrasing your question."

        # Extract SQL queries
        sql_queries = []
        for msg in result_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "execute_sql":
                        sql_queries.append(tc["args"].get("query", ""))
        last_sql = sql_queries[-1] if sql_queries else None

        # Try to get result data for charts
        table_data = None
        if last_sql:
            try:
                conn = sqlite3.connect(DB_PATH)
                import pandas as pd
                df = pd.read_sql_query(last_sql, conn)
                conn.close()
                if not df.empty and len(df.columns) >= 2:
                    table_data = {
                        "columns": df.columns.tolist(),
                        "rows": df.values.tolist()[:100],
                    }
            except Exception:
                pass

        # Save to MongoDB conversation
        now = datetime.utcnow()
        user_msg = {"role": "user", "content": question, "timestamp": now}
        asst_msg = {
            "role": "assistant",
            "content": final_answer,
            "sql": last_sql,
            "timestamp": now,
        }

        if convo_id:
            try:
                # Update title if this is the first message
                convo = conversations_col.find_one({"_id": ObjectId(convo_id)})
                update = {
                    "$push": {"messages": {"$each": [user_msg, asst_msg]}},
                    "$inc": {"message_count": 2},
                    "$set": {"updated_at": now},
                }
                if convo and convo.get("message_count", 0) == 0:
                    # Set title from first question (truncated)
                    title = question[:60] + ("..." if len(question) > 60 else "")
                    update["$set"]["title"] = title

                conversations_col.update_one(
                    {"_id": ObjectId(convo_id), "user_id": session["user_id"]},
                    update,
                )
            except Exception:
                pass

        return jsonify({
            "answer": final_answer,
            "sql": last_sql,
            "table_data": table_data,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

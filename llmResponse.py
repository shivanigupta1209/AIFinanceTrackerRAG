# llmResponse.py
import os
from dotenv import load_dotenv
import google.generativeai as genai  # <-- change this line

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)
def generate_sql_from_query(user_query, table_name="transactions"):
    """
    Use Gemini to translate natural language into SQL.
    We'll pass schema context so the model generates valid Supabase SQL.
    """
    schema_hint = """
Table: transactions
Columns: id, date, type, amount, status, userId, category, accountId
Example:
'how much did I spend this month on groceries?'
→ SELECT SUM(amount) AS total_spent
  FROM transactions
  WHERE type = 'EXPENSE' AND category ILIKE '%groceries%'
    AND date_trunc('month', date) = date_trunc('month', CURRENT_DATE);
    """

    prompt = f"""
You are an expert SQL translator.
Given the user's question and the schema below,
generate a single valid SQL query (PostgreSQL syntax) that answers it.
Generate exactly one valid SQL SELECT query (no semicolons, no multiple statements).
The query will be wrapped and filtered by user/account automatically — do not add userId or accountId filters yourself.

User question: "{user_query}"
{schema_hint}

Return only the SQL query (no explanation).
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def classify_query_intent(user_query: str) -> str:
    """
    Use Gemini to classify the intent of a query as either:
    - 'analytical'  (requires SQL operations like sum, count, average, filter)
    - 'semantic'    (textual lookup, summarization, natural info retrieval)
    """
    prompt = f"""
Classify the user's intent into one of two categories:
1. "analytical" → questions involving totals, counts, averages, date ranges, numeric filters, or SQL-like operations.
2. "semantic" → questions asking for explanations, text summaries, or specific transaction lookups.

User query: "{user_query}"

Respond with only one word: analytical OR semantic.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    intent = response.text.strip().lower()

    if "analytical" in intent:
        return "analytical"
    return "semantic"


def build_context_from_records(records):
    """
    Build a context string from the list of records fetched from the DB.
    Each record should have metadata with columns.
    """
    if not records:
        return "No relevant records found."
    
    lines = ["RESPONSE:"]
    for rec in records:
        cols = rec.get("metadata", {}).get("columns", {})
        date = cols.get("date", "?")
        category = cols.get("category", "?")
        amount = cols.get("amount", "?")
        type_ = cols.get("type", "?")
        if type_ == "EXPENSE":
            lines.append(f"- {date}: {category}, ${amount}")
    return "\n".join(lines)

def get_llm_answer(user_query, records):
    """
    Generate LLM answer based on user query and records.
    """
    context = build_context_from_records(records)
    prompt = f"""
User asked: "{user_query}"
{context}
Please answer the user's question based on this data.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)

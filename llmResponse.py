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


# def build_context_from_records(records):
#     """
#     Build a context string from the list of records fetched from the DB.
#     Each record should have metadata with columns.
#     """
#     if not records:
#         return "No relevant records found."
    
#     lines = ["RESPONSE:"]
#     for rec in records:
#         cols = rec.get("metadata", {}).get("columns", {})
#         date = cols.get("date", "?")
#         category = cols.get("category", "?")
#         amount = cols.get("amount", "?")
#         type_ = cols.get("type", "?")
#         if type_ == "EXPENSE":
#             lines.append(f"- {date}: {category}, ${amount}")
#     return "\n".join(lines)
def build_context_from_records(records):
    """
    Convert list of record dictionaries into a clean, readable text format for the LLM.
    """
    if not records:
        return "No records were found."

    if isinstance(records, dict):
        records = [records]

    # Make a readable text version (like CSV-style)
    context_lines = []
    for i, rec in enumerate(records, 1):
        if isinstance(rec, dict):
            line = ", ".join(f"{k}: {v}" for k, v in rec.items())
        else:
            line = str(rec)
        context_lines.append(f"Record {i}: {line}")

    return "\n".join(context_lines)


def get_llm_answer(user_query, records):
    """
    Generate LLM answer based on user query and records.
    This prompt ensures Gemini uses the given data instead of refusing.
    """
    context = build_context_from_records(records)

    prompt = f"""
You are a helpful data analysis assistant.

Below is a set of records retrieved from a financial transactions database.
Use ONLY this data to answer the user's question.

User question:
{user_query}

Database records:
{context}

If the records show totals, averages, or other numeric summaries, 
report them directly and clearly in natural language.
Do not refuse to answer. Do not mention that you lack access to finances.
Answer concisely and factually based on the given data.

For standard queries like hi or hello, give the response as show below:
'hello': 'Hello! How can I help you today?',
'hi': 'Hi there! What would you like to know?',
'how are you': 'I am just a bot, but I am functioning as expected! How can I assist you?',
'help': 'You can ask me questions about NotebookLM, its features, or just say hello!',
'bye': 'Goodbye! Feel free to come back if you have more questions.',
'thanks': 'You\'re welcome! Let me know if there\'s anything else I can help with.',
'thank you': 'You\'re welcome! Let me know if there\'s anything else I can help with.'
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else str(response)
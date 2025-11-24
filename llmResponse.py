import os
from dotenv import load_dotenv
import google.generativeai as genai 

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
    query = user_query.lower()

    # Keywords that indicate semantic comparison reasoning,
    # but still should go to RAG (not SQL)
    comparative_triggers = [
        "compare", "difference", "versus", "vs",
        "increase", "decrease", "why did", "most", "least",
        "higher", "lower", "trend"
    ]

    analytical_keywords = [
        "total", "sum", "average", "count", "how much",
        "spent on", "per month", "per day", "filter",
        "greater than", "less than"
    ]

    # If clearly analytical → return analytical
    if any(k in query for k in analytical_keywords):
        return "analytical"

    # Comparative or semantic → return semantic
    return "semantic"

def build_context_from_records(records):
    """
    Convert list of record dictionaries into a clean, readable text format for the LLM.
    """
    if not records:
        return "No records were found."

    if isinstance(records, dict):
        records = [records]

    context_lines = []
    for i, rec in enumerate(records, 1):
        if isinstance(rec, dict):
            line = ", ".join(f"{k}: {v}" for k, v in rec.items())
        else:
            line = str(rec)
        context_lines.append(f"Record {i}: {line}")

    return "\n".join(context_lines)
converation_history = [] 
def get_llm_answer(user_query, records):
    global converation_history
    """
    Generate an intelligent and context-aware answer based on the user's query and retrieved records.
    Includes friendly responses for greetings and polite messages,
    while responsibly handling financial advice.
    """
    trimmed_history = converation_history[-5:] 
    context = build_context_from_records(records)

    prompt = f"""
    Conversation History(last 5 exchanges):
    {trimmed_history}
    
    You are a helpful and precise financial assistant. You MUST answer using ONLY the records provided below.
    Never say "I don't have access to data". If required, politely ask the user to clarify their question.

    Your responsibilities:
    1. **Greet normally** to casual messages (“hello”, “how are you”, etc.)
    2. **Analytical reasoning**:
    - If the question involves comparing time periods, detect which records belong to each period.
    - Compute totals per category, per month, per account, or per merchant.
    - Identify trends: increases, decreases, anomalies, or unusually high spending.
    - Identify “top spending categories” by summing amounts.
    - For “why did my spending increase?”, analyze differences and explain the main contributing categories.

    3. **Semantic reasoning**:
    - Summarize the pattern visible in the records.
    - Provide insights based on spending descriptions and categories.

    4. **Tone & Safety**:
    - Keep the tone natural, warm, and clear.
    - DO NOT give financial advice like investment instructions.
    - DO NOT use disclaimers unless the user asks for actual financial recommendations.

    5. **If context is insufficient**:
    - NEVER say "I have no data".
    - Instead say: “Could you clarify the month or category you’re referring to?” or "Could you please clarify your question?".

    ---

    ### User Query:
    {user_query}

    ### Retrieved Records:
    {context}

    Now generate the best possible answer using only the details in these records.
    If the question is comparative, compute the necessary totals and clearly explain the reasoning.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    #return response.text.strip() if hasattr(response, "text") else str(response)
    answer = response.text.strip() if hasattr(response, "text") else str(response)

    # for conversation memory
    conversation_history.append({"user": user_query, "assistant": answer})

    return answer
import os
from dotenv import load_dotenv
import google.generativeai as genai 

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)
def generate_sql_from_query(user_query, table_name="transactions"):
    schema_hint = """
You are generating SQL for the following PostgreSQL table:

Table: transactions
Columns:
- id (text)
- type (TransactionType ENUM)- 'INCOME' or 'EXPENSE'
- amount (numeric)
- description (text)
- date (timestamp)
- category (text): food, housing, groceries, entertainment, utilities, transportation, healthcare, education, personal care, travel, gifts & donations, insurance, bills & fees, investments, other expenses
- receiptUrl (text)
- isRecurring (boolean)
- recurringInterval (RecurringInterval ENUM)
- nextRecurringDate (timestamp)
- lastProcessed (timestamp)
- status (TransactionStatus ENUM)
- userId (text)
- accountId (text)
- createdAt (timestamp)
- updatedAt (timestamp)

IMPORTANT RULES:
1. Generate ONLY ONE SQL SELECT query.
2. Do NOT output semicolons.
3. Do NOT include userId or accountId filters. They will be added automatically by the backend.
4. Use ONLY the columns listed above. Do NOT invent columns.
5. Use valid PostgreSQL syntax only.
6. Use ILIKE for text/category matching.
7. For EXPENSE queries, assume type = 'EXPENSE' unless the user explicitly asks otherwise.
8. For category search: category ILIKE '%keyword%'.
9. For timestamp/date logic, follow these rules:

   • If the user asks for a specific date:
       Use a date range:
       date >= 'YYYY-MM-DD'
       AND date < 'YYYY-MM-DD'::date + INTERVAL '1 day'

   • If the user asks for a specific month (e.g., "September 2024"):
       Use:
       date >= 'YYYY-09-01'
       AND date <  'YYYY-10-01'

   • If the user asks for a month WITHOUT specifying year:
       Use:
       EXTRACT(MONTH FROM date) = <month_number>

   • If the user asks for "this month":
       Use:
       date_trunc('month', date) = date_trunc('month', CURRENT_DATE)

10. For totals, alias properly:
    SUM(amount) AS total_spent
    SUM(amount) AS total_income
    etc.

11. When asked about "how much was spent", assume:
    type = 'EXPENSE'.

12. If the query involves comparison, differences, trends, ranking, “why”, or any comparative trigger listed below, 
    DO NOT attempt to write a comparative SQL query. 
    Instead, return a broad SELECT * query so the LLM can analyze the time periods itself.

    Comparative triggers include:
      "compare", "difference", "versus", "vs",
      "increase", "decrease", "why did", "most", "least",
      "higher", "lower", "trend"

    Examples:
    Q: "Why did my spending increase in September compared to October?"
    → SELECT * FROM transactions

    Q: "What category did I spend the most money on last month?"
    → SELECT * FROM transactions

    Q: "Compare my grocery spending this month vs last month"
    → SELECT * FROM transactions WHERE category ILIKE '%grocery%' 

EXAMPLES:

Example 1:
Q: "How much did I spend this month on groceries?"
→ SELECT SUM(amount) AS total_spent
    FROM transactions
    WHERE type = 'EXPENSE'
      AND category ILIKE '%groceries%'
      AND date_trunc('month', date) = date_trunc('month', CURRENT_DATE)

Example 2:
Q: "How much money was spent in September?"
→ SELECT SUM(amount) AS total_spent
    FROM transactions
    WHERE type = 'EXPENSE'
      AND EXTRACT(MONTH FROM date) = 9

Example 3:
Q: "Show my expenses on 2025-09-27 or what is the total amount of money spent on september 27 this year?"
→ SELECT *
    FROM transactions
    WHERE type = 'EXPENSE'
      AND date >= '2025-09-27'
      AND date < '2025-09-28'

Example 4:
Q: "Why did my spending increase in September compared to October?"
→ SELECT * FROM transactions

Example 5:
Q: "how much did i spend on goa trip?"
→ SELECT SUM(amount) AS total_spent
    FROM transactions
    WHERE type = 'EXPENSE'
      AND description ILIKE '%goa%'


"""

    prompt = f"""
    You are an expert AI that converts natural language into SQL.

    User question: "{user_query}"

    Using the schema and rules below, return ONLY a valid SQL SELECT query.
    Provide no explanations or extra text.

    {schema_hint}
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
    if any(k in query for k in (analytical_keywords + comparative_triggers)):
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
conversation_history = [] 
# def get_llm_answer(user_query, records):
#     global conversation_history
#     """
#     Generate an intelligent and context-aware answer based on the user's query and retrieved records.
#     Includes friendly responses for greetings and polite messages,
#     while responsibly handling financial advice.
#     """
#     trimmed_history = conversation_history[-5:] 
#     context = build_context_from_records(records)

#     prompt = f"""
#     Conversation History(last 5 exchanges):
#     {trimmed_history}

#     You are a helpful and precise financial assistant. You MUST answer using ONLY the records provided below.
#     Respond in plain text only. Do NOT use Markdown formatting, bullets, asterisks (*), dashes (-), plus signs (+), or code fences.
#     Never say "I don't have access to data". If required, politely ask the user to clarify their question.

#     Your responsibilities:
#     1. **Greet normally** to casual messages (“hello”, “how are you”, etc.)
#     2. **Analytical reasoning**:
#     - Answer questions about totals, sums, averages, counts, and filters.
#     - If the question involves comparing time periods, detect which records belong to each period.
#     - Compute totals per category, per month, per account, or per merchant.
#     - Identify trends: increases, decreases, anomalies, or unusually high spending.
#     - Identify “top spending categories” by summing amounts.
#     - For “why did my spending increase?”, analyze differences and explain the main contributing categories.

#     3. **Semantic reasoning**:
#     - Summarize the pattern visible in the records.
#     - Provide insights based on spending descriptions and categories.

#     4. **Tone & Safety**:
#     - Keep the tone natural, warm, and clear.
#     - DO NOT give financial advice like investment instructions.
#     - DO NOT use disclaimers unless the user asks for actual financial recommendations.

#     5. **If context is insufficient**:
#     - NEVER say "I have no data".
#     - Instead say: “Could you clarify the month or category you’re referring to?” or "Could you please clarify your question?".

#     Example query structure:
#     Q: "How much money was spent in September?"
#     → SELECT SUM(amount) AS total_spent
#         FROM transactions
#         WHERE type = 'EXPENSE'
#         AND EXTRACT(MONTH FROM date) = 9
#     Therefore, total_spent is $X based on the records provided.
#     ---

#     ### User Query:
#     {user_query}

#     ### Retrieved Records:
#     {context}

#     Now generate the best possible answer using only the details in these records.
#     If the question is comparative, compute the necessary totals and clearly explain the reasoning.
#     """

#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     #return response.text.strip() if hasattr(response, "text") else str(response)
#     answer = response.text.strip() if hasattr(response, "text") else str(response)

#     # for conversation memory
#     conversation_history.append({"user": user_query, "assistant": answer})

#     return answer
def get_llm_answer(user_query, records):
    global conversation_history

    trimmed_history = conversation_history[-5:] 
    context = build_context_from_records(records)

    prompt = f"""
You are a reliable financial assistant. Your answers must always be clear, correct, and based ONLY on the records provided.
NEVER return an empty answer.
NEVER say "I don't have data" or "I don't have access to data."
If the user’s question is unclear, politely ask for clarification.

IMPORTANT FORMAT RULES:
- Respond in plain text only.
- Do NOT use Markdown formatting.
- Do NOT use bullets, asterisks (*), dashes (-), plus signs (+), or code fences.
- Write in short, natural sentences.

CONVERSATION HISTORY (last 5 messages):
{trimmed_history}

YOUR SKILLS:
1. Greeting queries:
   Respond warmly and simply to casual greetings like "hello", "hi", or "how are you".

2. Analytical reasoning:
   Use the records to compute totals, sums, averages, counts, or filters.
   If the question involves a month, category, or date range, identify those records and calculate the results.
   If the question compares two periods (like September vs October), compute totals for EACH period separately, then compare them.
   If the question asks "why did my spending increase", identify which categories contributed most to the increase or decrease.
   If the question asks "on what did I spend the most", group transactions by category and find the largest total.

3. Semantic reasoning:
   Summarize patterns or spending behavior visible in the records.
   Identify trends or notable changes in the data.

4. If context is insufficient:
   Ask for clarification politely, such as:
   "Could you clarify the month or category you're referring to?"

Example query structure:
    Q: "How much money was spent in September?"
    → SELECT SUM(amount) AS total_spent
        FROM transactions
        WHERE type = 'EXPENSE'
        AND EXTRACT(MONTH FROM date) = 9
    Therefore, total_spent is $X based on the records provided.
    ---
USER QUESTION:
{user_query}

RECORDS YOU MUST USE:
{context}

Now produce the best possible answer using only these records.
If the question is comparative, compute totals for each period and explain the difference clearly.
Never leave the answer blank.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    answer = response.text.strip() if hasattr(response, "text") else str(response)

    # safety: never return empty answer
    if not answer or not answer.strip():
        answer = "I can help with that. Could you clarify your question a bit?"

    conversation_history.append({"user": user_query, "assistant": answer})
    return answer

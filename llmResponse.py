# llmResponse.py
import os
from dotenv import load_dotenv
import google.generativeai as genai  # <-- change this line

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

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

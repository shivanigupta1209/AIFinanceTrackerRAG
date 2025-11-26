# retrieve.py
import json
import os
from fastapi import FastAPI, Request
from supabase import create_client, Client
from dotenv import load_dotenv
import numpy as np
from embeddingCreation import get_gemini_embedding  
from llmResponse import get_llm_answer, build_context_from_records, classify_query_intent, generate_sql_from_query  # your LLM function

load_dotenv()
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not found in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="RAG Retrieval API")

def _sanitize_sql(sql: str) -> str:
    """Remove markdown fences, language tags, trailing semicolon and whitespace."""
    if not sql:
        return ""
    s = sql.strip()
    s = s.replace("```sql", "").replace("```", "")
    s = s.strip("` \n\r\t")
    s = s.rstrip().rstrip(";")
    return s.strip()

def match_documents_online(query_embedding, userId, accountId, top_k=5):
    """
    Query Supabase (pgvector) for top-K embeddings matching query_embedding,
    filtered by userId and accountId.
    """
    # np array to pgvector list
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    emb_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"

    
    res = supabase.rpc(
        "match_embeddings",
        {
            "query_embedding": query_embedding,
            "user_id": userId,
            "account_id": accountId,
            "top_k": top_k
        }
    ).execute()

    if hasattr(res, "error") and res.error:
        raise Exception(f"Supabase RPC error: {res.error}")
    if isinstance(res, dict) and "error" in res and res["error"]:
        raise Exception(f"Supabase RPC error: {res['error']}")
    data = res.data
    return data

# # FastAPI endpoint
# # ------------ FIX: PERIOD-AWARE SEMANTIC FETCHING ------------
# def semantic_period_fetch(query: str, user_id: str, account_id: str):
#     """
#     Detect periods like 'September', 'October', 'last month', etc.
#     Fetch ALL records for those periods instead of using vector embeddings.
#     """
#     userid = user_id
#     accountid = account_id
#     # payload = {
#     #    "query": sql_query,
#     #    "user_id": userid,
#     #    "account_id": accountid
#     # }
#     # print("Calling execute_sql with payload:", payload)
#     # exec_res = supabase.rpc("execute_sql_wrapper", payload).execute()


#     text = query.lower()

#     # (1) Detect specific months
#     month_map = {
#         "january": 1, "february": 2, "march": 3, "april": 4,
#         "may": 5, "june": 6, "july": 7, "august": 8,
#         "september": 9, "october": 10, "november": 11, "december": 12
#     }

#     months_detected = [m for m in month_map if m in text]

#     records = []

#     if months_detected:
#         for m in months_detected:
#             month_num = month_map[m]
#                 #   AND "userId" = '{userid}'
#                 #   AND "accountId" = '{accountid}'
#             sql = f"""
#                 SELECT *
#                 FROM transactions
#                 WHERE EXTRACT(MONTH FROM date) = {month_num}
#                 ORDER BY date;
#             """
#             payload = {
#                 "query": sql,
#                 "user_id": userid,
#                 "account_id": accountid
#             }
#             res = supabase.rpc("execute_sql_wrapper", payload).execute()

#             if res.data:
#                 if isinstance(res.data, str):
#                     rows = json.loads(res.data)
#                 else:
#                     rows = res.data
#                 records.extend(rows)

#         return records

#     # (2) If “this month”
#     if "this month" in text:
#         sql = """
#             SELECT *
#             FROM transactions
#             WHERE date_trunc('month', date) = date_trunc('month', CURRENT_DATE)
#             ORDER BY date;
#         """
#         # AND "userId" = '{user_id}'
#         #       AND "accountId" = '{account_id}'
#         payload = {
#             "query": sql,
#             "user_id": userid,
#             "account_id": accountid
#         }
#         res = supabase.rpc("execute_sql_wrapper", payload).execute()

#         return json.loads(res.data) if isinstance(res.data, str) else res.data

#     # fallback → normal vector-based semantic search
#     return None
def generate_period_sql(user_query):
    """
    Uses Gemini to generate SQL for month-based or period-based semantic queries.
    ALWAYS returns ONE SQL SELECT * query that retrieves ALL relevant raw rows.
    """

    period_schema_hint = """
You are generating SQL for retrieving RAW transaction rows for time-based queries.

Table: transactions

RULES:
1. ALWAYS return: SELECT * FROM transactions ...
2. NEVER return aggregates like SUM, COUNT, AVG.
3. NEVER include userId or accountId filters.
4. Use PostgreSQL syntax only.

5. If the query mentions specific months such as:
   "September and October"
   "compare September and October"
   → Detect both months and RETURN ONE SQL QUERY using:
     EXTRACT(MONTH FROM date) IN (<month_numbers>)

   Example:
   SELECT * FROM transactions
   WHERE EXTRACT(MONTH FROM date) IN (9, 10)
   ORDER BY date;

6. If the query mentions only one month:
   SELECT * FROM transactions WHERE EXTRACT(MONTH FROM date) = <month_number>

7. If the query says “this month”:
   date_trunc('month', date) = date_trunc('month', CURRENT_DATE)

8. If the query says “last month”:
   date_trunc('month', date) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month')

9. Output ONLY a SQL query string. No JSON. No explanation.
"""

    prompt = f"""
You are an expert SQL generator.

User query: "{user_query}"

Generate exactly ONE SQL SELECT query following the rules below:
{period_schema_hint}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def semantic_period_fetch(query, user_id, account_id):
    sql = generate_period_sql(query)

    if not sql:
        return None  # Not a period-based query
    userid = user_id
    accountid = account_id
    payload = {
        "query": sql,
        "user_id": userid,
        "account_id": accountid
    }
    # Run SQL using your RPC wrapper (which auto-adds userId/accountId filters)
    res = supabase.rpc("execute_sql_wrapper",payload).execute()

    rows = json.loads(res.data) if isinstance(res.data, str) else res.data

    # Tag rows with month number for LLM clarity
    for r in rows:
        try:
            month_num = int(r["date"][5:7])  # extract month from string timestamp
            r["__month"] = month_num
        except:
            pass

    return rows

@app.post("/retrieve")
async def retrieve(request: Request):
    data = await request.json()
    query = data.get("query")
    user_id = data.get("userid", "2896d2d5-915e-463b-85c5-fe1dcd141486")
    account_id = data.get("accountid", "ba67685c-4878-4d5c-bb0f-75bcdb4c763b")
    top_k = data.get("top_k", 5)

    if not query or not user_id or not account_id:
        return {"status": "Missing required fields: query, userId, accountId"}

    try:
        
        intent = classify_query_intent(query)
        print(f"Detected intent: {intent}")

        
        if intent == "analytical":
            sql_query = generate_sql_from_query(query, table_name="transactions")

            print("Generated SQL:", sql_query)

            sql_query = _sanitize_sql(sql_query)

            
            print("Sanitized SQL:", sql_query)
            if not sql_query.lower().startswith("select"):
                raise ValueError("Only SELECT queries are allowed.")
            userid = user_id
            accountid = account_id
            payload = {
                "query": sql_query,
                "user_id": userid,
                "account_id": accountid
            }
            print("Calling execute_sql with payload:", payload)
            exec_res = supabase.rpc("execute_sql_wrapper", payload).execute()

            
            if isinstance(exec_res, dict):
                err = exec_res.get("error")
                data = exec_res.get("data")
            else:
                err = getattr(exec_res, "error", None)
                data = getattr(exec_res, "data", None)

            if err:
                raise Exception(f"Supabase execute_sql RPC error: {err}")
            
            result_rows = []
            if data is None:
                result_rows = []
            elif isinstance(data, str):
                try:
                    result_rows = json.loads(data)
                except Exception:
                    result_rows = [data]
            elif isinstance(data, (list, tuple)):
                result_rows = list(data)
            else:
                result_rows = [data]

            
            try:
                summary_text = build_context_from_records(result_rows)
            except Exception:
                
                summary_text = json.dumps(result_rows, default=str, indent=2)

            answer = ""
            try:
                answer = get_llm_answer(query, result_rows)
            except Exception as e:
                print("LLM call failed for analytical route:", e)
                answer = ""

            return {
                "mode": "analytical",
                "query": query,
                "sql_query": sql_query,
                "raw_result": result_rows,
                "answer": answer
            }
        else:
            # userid = user_id
            # accountid = account_id
            # period_results = semantic_period_fetch(query, userid, accountid)

            # # If periods detected → skip vector search entirely
            # if period_results:
            #     # Now give these full-month records to LLM
            #     print("detected intent: semantic-intent")
            #     answer = get_llm_answer(query, period_results)
            #     return {
            #         "mode": "semantic-period",
            #         "query": query,
            #         "records_used": period_results,
            #         "answer": answer
            #     }

            #Semantic route
            query_embedding = get_gemini_embedding(query, dim=384)
            top_docs = match_documents_online(query_embedding, user_id, account_id, top_k=top_k)
            answer = get_llm_answer(query, top_docs)

            print("llm answer: \n", answer)

            return {
                "mode": "semantic",
                "query": query,
                "answer": answer,
                "top_k_results": top_docs
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}
#Fpr local testin
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

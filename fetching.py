# retrieve.py
import os
from fastapi import FastAPI, Request
from supabase import create_client, Client
from dotenv import load_dotenv
import numpy as np
from embeddingCreation import get_gemini_embedding  # your embedding function
from llmResponse import get_llm_answer, build_context_from_records, classify_query_intent, generate_sql_from_query  # your LLM function
# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not found in .env")

# -------------------------------
# 2️⃣ Initialize Supabase client
# -------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# 3️⃣ Initialize FastAPI
# -------------------------------
app = FastAPI(title="RAG Retrieval API")

# -------------------------------
# 4️⃣ Helper function: match_documents_online
# -------------------------------

# res = supabase.table("embeddingsnew") \
#     .select("*") \
#     .eq("user_id", userId) \
#     .eq("account_id", accountId) \
#     .execute()
def _sanitize_sql(sql: str) -> str:
    """Remove markdown fences, language tags, trailing semicolon and whitespace."""
    if not sql:
        return ""
    s = sql.strip()
    # remove triple-backtick fences and optional language tag
    s = s.replace("```sql", "").replace("```", "")
    # remove any leading/trailing backticks left over
    s = s.strip("` \n\r\t")
    # remove trailing semicolon
    s = s.rstrip().rstrip(";")
    return s.strip()

def match_documents_online(query_embedding, userId, accountId, top_k=5):
    """
    Query Supabase (pgvector) for top-K embeddings matching query_embedding,
    filtered by userId and accountId.
    """
    # Convert numpy array to pgvector-compatible string
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    emb_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"

    # Raw SQL query using pgvector distance operator <=> (cosine distance)
    # sql = f"""
    #     SELECT *, embedding <=> '{emb_str}'::vector AS distance
    #     FROM embeddingsnew
    #     WHERE "user_id" = '{userId}' AND "account_id" = '{accountId}'
    #     ORDER BY distance
    #     LIMIT {top_k};
    # """

    # # Execute raw SQL
    # res = supabase.table("embeddingsnew") \
    # .select("*") \
    # .eq("user_id", userId) \
    # .eq("account_id", accountId) \
    # .execute()
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

# -------------------------------
# 5️⃣ FastAPI endpoint
# -------------------------------

@app.post("/retrieve")
async def retrieve(request: Request):
    data = await request.json()
    query = data.get("query")
    user_id = data.get("userid")
    account_id = data.get("accountid")
    top_k = data.get("top_k", 5)

    if not query or not user_id or not account_id:
        return {"status": "❌ Missing required fields: query, userId, accountId"}

    try:
        # Step 0️⃣: Let Gemini classify the intent
        intent = classify_query_intent(query)
        print(f"Detected intent: {intent}")

        # Step 1️⃣: Analytical route
        if intent == "analytical":
            sql_query = generate_sql_from_query(query, table_name="transactions")

            # Ensure filters for user/account
            # if "where" in sql_query.lower():
            #     sql_query += f" AND userId = '{userId}' AND accountId = '{accountId}'"
            # else:
            #     sql_query += f" WHERE userId = '{userId}' AND accountId = '{accountId}'"

            print("Generated SQL:", sql_query)

            # Execute query
            #sql_query = sql_query.strip().rstrip(';')
            sql_query = _sanitize_sql(sql_query)

            # debug print to verify cleaned SQL
            print("Sanitized SQL:", sql_query)
            if not sql_query.lower().startswith("select"):
                raise ValueError("Only SELECT queries are allowed.")
            userid = user_id
            accountid = account_id
            payload = {
                "query": sql_query,
                "userid": userid,
                "accountid": accountid,
            }

            result = supabase.rpc("execute_sql", payload).execute()
            if result.error:
                raise Exception(result.error)
            # return result.data
            # sql_response = supabase.rpc("execute_sql", {"query": sql_query}).execute()
            # result = sql_response.data if hasattr(sql_response, "data") else sql_response

            return {
                "mode": "analytical",
                "query": query,
                "sql_query": sql_query,
                "result": result
            }

        # Step 2️⃣: Semantic route
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
        return {"status": "❌ error", "error": str(e)}

# @app.post("/retrieve")
# async def retrieve(request: Request):
#     """
#     Input JSON: {"query": "text to search", "userId": "...", "accountId": "...", "top_k": 5}
#     Returns top-K matched embeddings from Supabase filtered by user/account.
#     """
#     data = await request.json()
#     query = data.get("query")
#     userId = data.get("userId")
#     accountId = data.get("accountId")
#     top_k = data.get("top_k", 5)

#     if not query or not userId or not accountId:
#         return {"status": "❌ Missing required fields: query, userId, accountId"}

#     try:
#         # 1️⃣ Embed the query
#         query_embedding = get_gemini_embedding(query, dim=384)

#         # 2️⃣ Get top-K embeddings from Supabase
#         top_docs = match_documents_online(query_embedding, userId, accountId, top_k=top_k)

#         records= {
#             "query": query,
#             "userId": userId,
#             "accountId": accountId,
#             "top_k_results": top_docs
#         }
#         answer = get_llm_answer(query, top_docs)
#         print("LLM Answer:", answer)
#         return records

    # except Exception as e:
    #     return {"status": "❌ error", "error": str(e)}

# -------------------------------
# 6️⃣ Run locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

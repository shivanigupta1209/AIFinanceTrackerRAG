# retrieve.py
import os
from fastapi import FastAPI, Request
from supabase import create_client, Client
from dotenv import load_dotenv
import numpy as np
from embeddingCreation import get_gemini_embedding  # your embedding function

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

    if res.error:
        raise Exception(f"Supabase query failed: {res.error}")
    return res.data

# -------------------------------
# 5️⃣ FastAPI endpoint
# -------------------------------
@app.post("/retrieve")
async def retrieve(request: Request):
    """
    Input JSON: {"query": "text to search", "userId": "...", "accountId": "...", "top_k": 5}
    Returns top-K matched embeddings from Supabase filtered by user/account.
    """
    data = await request.json()
    query = data.get("query")
    userId = data.get("userId")
    accountId = data.get("accountId")
    top_k = data.get("top_k", 5)

    if not query or not userId or not accountId:
        return {"status": "❌ Missing required fields: query, userId, accountId"}

    try:
        # 1️⃣ Embed the query
        query_embedding = get_gemini_embedding(query, dim=384)

        # 2️⃣ Get top-K embeddings from Supabase
        top_docs = match_documents_online(query_embedding, userId, accountId, top_k=top_k)

        return {
            "query": query,
            "userId": userId,
            "accountId": accountId,
            "top_k_results": top_docs
        }

    except Exception as e:
        return {"status": "❌ error", "error": str(e)}

# -------------------------------
# 6️⃣ Run locally
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

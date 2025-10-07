# worker.py
from fastapi import FastAPI, Request
import uvicorn
import os
from supabase import create_client
from embeddingCreation import embed_and_insert
from dotenv import load_dotenv

# 🔹 Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
print("Supabase URL:", SUPABASE_URL, "Supabase Key:", SUPABASE_KEY)

# 🔹 Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🔹 Initialize FastAPI
app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return {"status": "❌ failed to parse JSON"}
    print("Received payload:", payload)

    event_type = payload.get("type")  # INSERT, UPDATE, DELETE
    table_name = payload.get("table")
    row = payload.get("record")  # Supabase sends the full row
    old_row = payload.get("old_record")  # For UPDATE/DELETE events

    # if event_type != "INSERT" or not row:
    #     return {"status": f"⚠️ ignored event type {event_type}"}

    # source_id = row.get("id") or (old_row.get("id") if old_row else None)
    # if not source_id:
    #     return {"status": "⚠️ row missing 'id'"}

    # 🔹 Check only if this specific row already has an embedding
    if event_type == "INSERT":
        source_id = row.get("id")
        if not source_id:
            return {"status": "⚠️ missing id in insert"}
        existing = supabase.table("embeddingsnew").select("*").eq("source_id", source_id).execute()
        if existing.data and len(existing.data) > 0:
            return {"status": f"⚠️ embedding already exists for {source_id}"}
        # 🔹 Prepare text for embedding
        text = " ".join(str(v) for v in row.values() if v is not None)
        # 🔹 Call your embedding function
        try:
            embed_and_insert(table_name, row, text)
            return {"status": f"✅ embedding inserted for {source_id}"}
        except Exception as e:
            return {"status": f"❌ embedding failed for {source_id}", "error": str(e)}

    elif event_type == "UPDATE":
        new_row = payload.get("record")
        old_row = payload.get("old_record")
        row = new_row or old_row  # just in case

        if not row:
            return {"status": "⚠️ no row data for update"}

        source_id = row.get("id")
        if not source_id:
            return {"status": "⚠️ missing id in update"}

        # delete old embedding if exists
        supabase.table("embeddingsnew").delete().eq("source_id", source_id).execute()

        # re-create embedding
        text = " ".join(str(v) for v in new_row.values() if v is not None)
        embed_and_insert(table_name, new_row, text)

        print(f"♻️ Updated embedding for {source_id}")
        return {"status": f"♻️ updated embedding for {source_id}"}

    elif event_type == "DELETE":
        row = payload.get("record") or payload.get("old_record")
        if not row:
            return {"status": "⚠️ no row data for delete"}
        source_id = row.get("id")
        if not source_id:
            return {"status": "⚠️ missing id in delete"}
        supabase.table("embeddingsnew").delete().eq("source_id", source_id).execute()
        print(f"🗑️ Deleted embedding for {source_id}")
        return {"status": f"🗑️ deleted embedding for {source_id}"}
        
    else:
        return {"status": f"⚠️ unhandled event type {event_type}"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, Request
# import uvicorn

# app = FastAPI()

# # import your existing functions
# from test import embed_and_insert  

# @app.post("/webhook")
# async def webhook(request: Request):
#     payload = await request.json()
    
#     table_name = payload.get("table")
#     row = payload.get("record")  # same shape as Supabase row

#     if row:
#         text = " ".join(str(v) for v in row.values() if v is not None)
#         embed_and_insert(table_name, row, text)  # reuse your function
#         return {"status": "✅ embedding inserted"}
#     else:
#         return {"status": "⚠️ no row received"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

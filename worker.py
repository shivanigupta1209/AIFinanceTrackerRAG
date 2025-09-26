from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# import your existing functions
from your_existing_script import embed_and_insert  

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    
    table_name = payload.get("table")
    row = payload.get("record")  # same shape as Supabase row

    if row:
        text = " ".join(str(v) for v in row.values() if v is not None)
        embed_and_insert(table_name, row, text)  # reuse your function
        return {"status": "✅ embedding inserted"}
    else:
        return {"status": "⚠️ no row received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

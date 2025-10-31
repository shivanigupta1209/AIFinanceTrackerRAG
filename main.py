# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from worker import app as worker_app
from fetching import app as fetching_app

app = FastAPI(title="RAG Full Backend")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  
        "https://chat-bot-welth.vercel.app",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount the two sub-apps ---
app.mount("/webhook", worker_app)   # webhook endpoint: /webhook/webhook
app.mount("/api", fetching_app)     # retrieval endpoint: /api/retrieve

@app.get("/")
def root():
    return {"message": "Backend running successfully 🚀"}

# (Render will use this file as the entrypoint)

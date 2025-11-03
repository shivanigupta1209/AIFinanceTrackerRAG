# 🧠 AI Finance Tracker (RAG + Supabase + Gemini)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to analyze and retrieve financial data such as transactions, accounts, and budgets using natural language queries.  
It combines **Supabase**, **pgvector**, **Gemini Embeddings**, and **FastAPI** with an automated webhook workflow for real-time embedding updates.

---

## 🚀 Features

- 🔄 **Automatic Embedding Generation** — whenever a new row is added or updated in the database  
- 🧩 **Vector Search using pgvector** — for similarity-based retrieval  
- 🧠 **RAG Pipeline Integration** — LLM responses grounded in real financial data  
- ⚡ **Supabase Webhooks** — trigger worker updates automatically  
- 🌐 **Full-Stack Deployment** — FastAPI backend + React frontend (Vercel/Render)

---

## 🧱 Tech Stack

| Component | Technology |
|------------|-------------|
| **Database** | Supabase (PostgreSQL + pgvector) |
| **Backend** | FastAPI, Uvicorn |
| **Embedding Model** | Gemini Embedding 001 |
| **Frontend** | React + Vercel |
| **Deployment** | Render |

---

## ⚙️ Setup Steps

### ✅ Completed Workflow

1. **Enable pgvector** in Supabase  
```sql
   CREATE EXTENSION IF NOT EXISTS vector;
```

2. **Create Embedding Table** for `transactions`, `accounts`, and `budget`.

3. **Build a Python Worker** using **Gemini Embedding 001**.

4. **Set Up Supabase Webhooks**
   * Triggered on `INSERT` and `UPDATE` for relevant tables
   * Webhook points to your FastAPI worker endpoint

5. **Verify Table Operations**
   * Insert / Update / Delete all functioning correctly ✅
   * Embeddings automatically updated

6. **Create `match_documents()` Function** and IVFFlat Index
```sql
   CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

7. **Build Retrieval API**
   * Embed query → call `match_documents()` → return top K results

8. **Feed to LLM / UI**
   * Connect the retrieval results to your LLM pipeline or chatbot UI.

---

## 🧩 Example Flow

1. A new transaction is added → webhook fires → worker generates embedding
2. Embedding stored in the `embeddings` table
3. When user asks *"How much did I spend on groceries in October?"* →
   Query is embedded → top matches fetched via `match_documents()` →
   Context passed to LLM → Answer returned to frontend UI

---

## 📊 Deployment

* **Frontend:** Deployed on **Vercel**
* **Backend / Worker:** Deployed on **Render**
* Environment variables stored securely via project settings

---

## 🎯 Deliverables

* ✅ Functional RAG pipeline
* ✅ Gemini-based embeddings
* ✅ Real-time Supabase webhooks
* ✅ Deployed backend + frontend
* ✅ Evaluation & Demo complete

---

## 🧰 Future Scope

* Add user-level personalization for financial insights
* Implement caching for frequent queries
* Extend support for multiple data sources
* Enhance chat UI with context persistence

---

### 👩‍💻 Author

**Shivani Gupta**  
Pre-final year B.Tech student | Data Science & Machine Learning Enthusiast  
📘 Certified in Machine Learning and Advanced Python Programming

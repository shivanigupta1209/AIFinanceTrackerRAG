import google.generativeai as genai
from google.generativeai import types
from supabase import create_client
from dotenv import load_dotenv
import os

# 🔹 Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 🔹 Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🔹 Configure Gemini (new syntax — no Client() object)
genai.configure(api_key=GEMINI_API_KEY)

# Function to create embeddings with Gemini
def get_gemini_embedding(text, dim=384):
    try:
        result = genai.embed_content(
            model="gemini-embedding-001",   # correct model name
            content=text,
            task_type="retrieval_document", # recommended task type for RAG embeddings
            title="Embedding generation",
            output_dimensionality=dim       # if supported
        )
        if isinstance(result, dict) and "embedding" in result:
            emb = result["embedding"]
        elif hasattr(result, "embedding"):
            emb = result.embedding
        elif hasattr(result, "embeddings") and result.embeddings:
            emb = result.embeddings[0].values
        else:
            raise ValueError("Unexpected embedding format received from Gemini API.")

        return emb
        #return result["embedding"]  # returns list of floats
    except Exception as e:
        print(f"⚠️ Gemini embedding failed: {e}")
        return []

# Function to insert embeddings into Supabase
def embed_and_insert(source_table, row, text):
    try:
        emb = get_gemini_embedding(text, dim=384)
        if not emb:
            print(f"⚠️ Skipped embedding for {source_table} id {row['id']}")
            return

        emb_str = f"[{', '.join(str(x) for x in emb)}]"  # convert to pgvector format

        response = supabase.table("embeddingsnew").insert({
            "source_table": source_table,
            "source_id": row["id"],
            "user_id": row.get("userId"),
            "account_id": row.get("accountId"),
            "chunk_text": text,
            "metadata": {"columns": row},
            "embedding": emb_str
        }).execute()

        if hasattr(response, "error") and response.error:
            print(f"❌ Insert failed for {source_table} id {row['id']}: {response.error}")
        else:
            print(f"✅ Inserted embedding for {source_table} id {row['id']}")

    except Exception as e:
        print(f"⚠️ Embedding failed for {source_table} id {row['id']}: {e}")

# Example for one table
if __name__ == "__main__":
    table_names = ["transactions"]  # extend later
    for table_name in table_names:
        rows_response = supabase.table(table_name).select("*").execute()

        if hasattr(rows_response, "error") and rows_response.error:
            print(f"❌ Failed to fetch rows from {table_name}: {rows_response.error}")
            continue

        rows = rows_response.data
        print(f"📊 {len(rows)} rows fetched from {table_name}")

        for row in rows:
            text = " ".join(str(v) for v in row.values() if v is not None)
            embed_and_insert(table_name, row, text)

    print("Backfill complete ✅")

# # from google import genai
# # from google.genai import types
# import google.generativeai as genai
# from google.generativeai import types
# from supabase import create_client
# from dotenv import load_dotenv
# import os

# # 🔹 Load environment variables
# load_dotenv()

# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")   # use only in server script
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # 🔹 Initialize Supabase client
# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# # 🔹 Initialize Gemini client
# client = genai.Client(api_key=GEMINI_API_KEY)

# # Function to create embeddings with Gemini
# def get_gemini_embedding(text, dim=384):
#     result = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=text,
#         config=types.EmbedContentConfig(output_dimensionality=dim),
#     )
#     return result.embeddings[0].values   # returns float list

# # Function to insert embeddings into Supabase
# def embed_and_insert(source_table, row, text):
#     try:
#         emb = get_gemini_embedding(text, dim=384)  # Gemini output
#         emb_str = f"[{', '.join(str(x) for x in emb)}]"  # convert to pgvector format

#         response = supabase.table("embeddingsnew").insert({
#             "source_table": source_table,
#             "source_id": row["id"],
#             "user_id": row.get("userId"),
#             "account_id": row.get("accountId"),
#             "chunk_text": text,
#             "metadata": {"columns": row},
#             "embedding": emb_str
#         }).execute()

#         if hasattr(response, "error") and response.error:
#             print(f"❌ Insert failed for {source_table} id {row['id']}: {response.error}")
#         else:
#             print(f"✅ Inserted embedding for {source_table} id {row['id']}")

#     except Exception as e:
#         print(f"⚠️ Embedding failed for {source_table} id {row['id']}: {e}")

# # Example for one table
# if __name__ == "__main__":
#     table_names = ["transactions"]  # extend later
#     for table_name in table_names:
#         rows_response = supabase.table(table_name).select("*").execute()

#         if hasattr(rows_response, "error") and rows_response.error:
#             print(f"❌ Failed to fetch rows from {table_name}: {rows_response.error}")
#             continue

#         rows = rows_response.data
#         print(f"📊 {len(rows)} rows fetched from {table_name}")

#         for row in rows:
#             text = " ".join(str(v) for v in row.values() if v is not None)
#             embed_and_insert(table_name, row, text)

#     print("Backfill complete ✅")

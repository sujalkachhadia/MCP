import os
import shutil
import re
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

# --- CRITICAL CHANGE: Use PyMuPDFLoader instead of PyPDFLoader ---
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file")

app = FastAPI(title="Supabase RAG API")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str
    k: int = 5  # top K chunks

class QueryResponse(BaseModel):
    context: str

# --- HELPER FUNCTION: TEXT CLEANER ---
def clean_text(text: str) -> str:
    """
    Fixes common PDF formatting issues.
    """
    # 1. Replace NULL bytes
    text = text.replace("\x00", "")
    
    # 2. Fix hyphenated words broken across lines (e.g. "automa-\ntion")
    text = re.sub(r'-\n', '', text)
    
    # 3. Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # 4. Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF → Clean → Semantic Chunk → Store in Supabase
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    temp_path = f"temp_{file.filename}"

    try:
        # Step 1: Save PDF temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Load PDF using PyMuPDF (Cleaner extraction)
        loader = PyMuPDFLoader(temp_path)
        raw_docs = loader.load()

        # Step 3: Clean the text content
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_content = clean_text(doc.page_content)
            
            # Skip empty pages
            if len(cleaned_content) > 10: 
                doc.page_content = cleaned_content
                cleaned_docs.append(doc)

        if not cleaned_docs:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Step 4: Semantic Chunking
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        chunks = splitter.split_documents(cleaned_docs)

        # Step 5: Insert into Supabase
        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)

            supabase.table("documents").insert({
                "content": chunk.page_content,
                "metadata": {"source": file.filename},
                "embedding": vector
            }).execute()

        return {
            "message": "PDF processed successfully",
            "chunks_stored": len(chunks)
        }

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query_db(request: QueryRequest):
    
    try:
        # 1. Generate embedding for the user query
        query_embedding = embeddings.embed_query(request.query)

        # 2. RPC parameters
        rpc_params = {
            "filter": {},
            "match_count": request.k,
            "match_threshold": 0.0,
            "query_embedding": query_embedding
        }

        # 3. Call Supabase
        rpc_builder = supabase.rpc("match_documents", rpc_params)
        rpc = rpc_builder.execute()

        if not rpc.data:
            return {"context": "No relevant documents found."}

        # 4. Build clean context text
        parts = []
        for row in rpc.data:
            # Clean again just in case database has leftover dirty data
            clean_chunk = row['content'].replace("\n", " ").strip()
            parts.append(f"--- INFO ---\n{clean_chunk}")
            
        context = "\n\n".join(parts)

        return {"context": context}

    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
import os
import shutil
import re
from typing import List, Optional
from datetime import datetime, timedelta

# --- IMPORTS FOR AUTH ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from supabase import create_client, Client
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "unsafe_default_key") # Change this in .env
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Simple hardcoded admin credentials for this example
ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "password")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file")

app = FastAPI(title="Supabase RAG API")

# --- AUTH SETUP ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- SUPABASE & AI SETUP ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- MODELS ---
class Token(BaseModel):
    access_token: str
    token_type: str

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    context: str

# --- AUTH FUNCTIONS ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# --- HELPER: TEXT CLEANER ---
def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- ENDPOINTS ---

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real app, you would check a database here.
    # We are checking against env variables for simplicity.
    if form_data.username != ADMIN_USER or form_data.password != ADMIN_PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    current_user: str = Depends(get_current_user) # PROTECTED
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyMuPDFLoader(temp_path)
        raw_docs = loader.load()

        cleaned_docs = []
        for doc in raw_docs:
            cleaned_content = clean_text(doc.page_content)
            if len(cleaned_content) > 10: 
                doc.page_content = cleaned_content
                cleaned_docs.append(doc)

        if not cleaned_docs:
            raise HTTPException(status_code=400, detail="Could not extract text")

        splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
        chunks = splitter.split_documents(cleaned_docs)

        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)
            supabase.table("documents").insert({
                "content": chunk.page_content,
                "metadata": {"source": file.filename},
                "embedding": vector
            }).execute()

        return {"message": "PDF processed successfully", "chunks_stored": len(chunks)}

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query_db(
    request: QueryRequest, 
    current_user: str = Depends(get_current_user) # PROTECTED
):
    try:
        query_embedding = embeddings.embed_query(request.query)
        rpc_params = {
            "filter": {},
            "match_count": request.k,
            "match_threshold": 0.0,
            "query_embedding": query_embedding
        }
        
        rpc = supabase.rpc("match_documents", rpc_params).execute()

        if not rpc.data:
            return {"context": "No relevant documents found."}

        parts = []
        for row in rpc.data:
            clean_chunk = row['content'].replace("\n", " ").strip()
            parts.append(f"--- INFO ---\n{clean_chunk}")
            
        context = "\n\n".join(parts)
        return {"context": context}

    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
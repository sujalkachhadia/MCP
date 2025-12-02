import requests
import json
import os
import re
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8080")

# Credentials for the MCP server to authenticate with the RAG API
RAG_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
RAG_PASSWORD = os.getenv("ADMIN_PASSWORD", "securepassword123")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

mcp = FastMCP("Abhi's Assistant")

# --- HELPER: GET AUTH TOKEN ---
def get_auth_token():
    """Authenticates with the RAG API and returns a JWT token."""
    url = f"{RAG_SERVER_URL.rstrip('/')}/token"
    payload = {
        "username": RAG_USERNAME,
        "password": RAG_PASSWORD
    }
    try:
        # FastAPI OAuth2 expects form-data, not JSON
        res = requests.post(url, data=payload)
        res.raise_for_status()
        return res.json().get("access_token")
    except Exception as e:
        print(f"Authentication Failed: {e}")
        return None

# --- TOOL 1: WEATHER ---
@mcp.tool()
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Fetch live weather for a city."""
    api_unit = "metric" if unit.lower() == "celsius" else "imperial"
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": OPENWEATHER_API_KEY, "units": api_unit}

    try:
        res = requests.get(base_url, params=params)
        res.raise_for_status()
        data = res.json()
        return (f"Weather in {location.title()}: {data['main']['temp']}° "
                f"({data['weather'][0]['description']})")
    except Exception as e:
        return f"Weather API Error: {e}"

# --- TOOL 2: RAG ---
@mcp.tool()
def query_pdf_knowledge(query: str, k: int = 5) -> str:
    """Send query to RAG server and clean the output."""
    
    # 1. Get Token
    token = get_auth_token()
    if not token:
        return "Error: Could not authenticate with Knowledge Base server."

    base_url = RAG_SERVER_URL.rstrip("/")
    url = f"{base_url}/query"
    
    # 2. Add Header
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        res = requests.post(url, json={"query": query, "k": k}, headers=headers)
        res.raise_for_status()
        
        raw_context = res.json().get("context", "")
        clean_context = re.sub(r'\s+', ' ', raw_context).strip()
        
        return clean_context if clean_context else "No info found."
        
    except Exception as e:
        return f"RAG Connection Error: {e}"

# --- MAIN HANDLER ---
@mcp.tool()
async def handle_message(message: str) -> str:
    msg = message.lower().strip()

    # 1. Check Weather
    if "weather" in msg:
        words = msg.split()
        city = words[-1] if "in" in words else "London" 
        return get_current_weather(city)

    # 2. Get Knowledge
    context = query_pdf_knowledge(message)
    if "Error" in context:
        return context

    # 3. Generate Answer
    prompt_text = f"""
    You are a helpful assistant. Answer the question using ONLY the context provided.
    
    CONTEXT:
    {context}
    
    QUESTION: 
    {message}
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
    chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()

    try:
        return await chain.ainvoke({})
    except Exception as e:
        return f"LLM Error: {e}"

if __name__ == "__main__":
    mcp.run(transport="sse")
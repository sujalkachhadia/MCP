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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8080")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

mcp = FastMCP("Abhi's Assistant")

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
    base_url = RAG_SERVER_URL.rstrip("/")
    url = f"{base_url}/query"
    
    try:
        res = requests.post(url, json={"query": query, "k": k})
        res.raise_for_status()
        
        raw_context = res.json().get("context", "")
        
        # Extra cleaning for the AI
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
        # Simple extraction logic
        words = msg.split()
        city = words[-1] if "in" in words else "London" # Fallback
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
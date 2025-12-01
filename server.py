import requests
import json
import os
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from dotenv import load_dotenv  # <--- CRITICAL IMPORT

# LangChain Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================
# CONFIGURATION
# ============================

# 1. Load environment variables from .env file
load_dotenv() 

# 2. Get Keys securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# 3. RAG Server URL
# Defaults to localhost (127.0.0.1) if not set (for local testing)
# Docker will override this to "http://rag-api:8080" via the .env file
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8080")

# 4. Validate Keys
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

mcp = FastMCP("Abhi's Assistant")


# ======================================================
# TOOL 1: WEATHER LOOKUP
# ======================================================
@mcp.tool()
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Fetch live weather for a city using OpenWeatherMap API.
    Returns a clean, human-readable weather summary.
    """

    api_unit = "metric" if unit.lower() == "celsius" else "imperial"
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units": api_unit
    }

    try:
        res = requests.get(base_url, params=params)
        res.raise_for_status()

        data = res.json()

        # Extract useful information
        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        desc = data["weather"][0]["description"].title()
        humidity = data["main"]["humidity"]

        return (
            f"Weather in {location.title()}:\n"
            f"Temperature: {temp}°{'C' if api_unit=='metric' else 'F'}\n"
            f"Feels Like: {feels}°\n"
            f"Condition: {desc}\n"
            f"Humidity: {humidity}%"
        )

    except requests.exceptions.HTTPError as e:
        return f"Weather API HTTP Error: {e}"
    except Exception as e:
        return f"Weather API Connection Error: {e}"

# ======================================================
# TOOL 2: QUERY SUPABASE RAG SERVER (The one on 8080)
# ======================================================
@mcp.tool()
def query_pdf_knowledge(query: str, k: int = 5) -> str:
    """
    Send the query to your FastAPI + Supabase RAG server (on 8080).
    Returns semantic chunks (NOT final answer).
    """
    # Remove trailing slash if present to avoid double slash
    base_url = RAG_SERVER_URL.rstrip("/")
    url = f"{base_url}/query"
    
    payload = {"query": query, "k": k}

    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return res.json().get("context", "No context returned")
    except Exception as e:
        # This will catch connection errors or HTTP errors from the RAG server
        return f"RAG Server Error: Could not connect to {url}. Details: {e}"


# ======================================================
# MAIN HANDLER (LLM + CONTEXT)
# ======================================================

@mcp.tool()
async def handle_message(message: str) -> str:
    """
    Intelligent Router:
    - If user asks about weather → call get_current_weather
    - If user asks summary → create summary using chunks + Gemini
    - Else → normal RAG QA
    """

    msg = message.lower().strip()

    # ------------------------------------------------
    # STEP 1 — WEATHER INTENT DETECTION
    # ------------------------------------------------
    weather_keywords = ["weather", "temperature", "climate", "forecast"]
    if any(word in msg for word in weather_keywords):

        # Extract city name properly
        city = None
        words = msg.replace("?", "").split()

        # Simple extraction: after "in"
        for i, w in enumerate(words):
            if w == "in" and i + 1 < len(words):
                city = words[i + 1]
                break

        if not city:
            return "Please specify a city name. Example: 'weather in London'"

        try:
            weather = get_current_weather(city)
            return weather
        except Exception as e:
            return f"Weather Tool Error: {e}"

    # ------------------------------------------------
    # STEP 2 — CALL RAG SERVER (Used for PDF Tasks)
    # ------------------------------------------------
    rag_response = query_pdf_knowledge(message)

    if "RAG Server Error" in rag_response:
        return rag_response

    # ------------------------------------------------
    # STEP 3 — SUMMARY REQUEST ROUTING
    # ------------------------------------------------
    summary_keywords = ["summary", "summarize", "overview"]
    if any(k in msg for k in summary_keywords):
        prompt_text = f"""
You are an expert summarizer.
Write a clear, concise 6–7 line summary based ONLY on the document context below.

DOCUMENT CONTEXT:
{rag_response}

RULES:
- Do NOT show chunks.
- Focus only on useful points.
"""
    else:
        # ------------------------------------------------
        # NORMAL RAG QA MODE
        # ------------------------------------------------
        prompt_text = f"""
You are an expert assistant. 
Use ONLY the context below to answer the user question.

CONTEXT:
{rag_response}

QUESTION:
{message}

RULES:
- Do NOT show chunks.
- Answer clearly in 4–5 lines.
"""

    # ------------------------------------------------
    # STEP 4 — Build LLM Chain
    # ------------------------------------------------
    prompt = ChatPromptTemplate.from_template(prompt_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",    # ♥ Latest Model
        google_api_key=GEMINI_API_KEY
    )

    chain = prompt | llm | StrOutputParser()

    # ------------------------------------------------
    # STEP 5 — Run chain safely
    # ------------------------------------------------
    try:
        final_answer = await chain.ainvoke({})
        return final_answer
    except Exception as e:
        return f"LLM Error: {e}"

# ======================================================
# MCP SERVER START
# ======================================================
if __name__ == "__main__":
    
    print("Starting MCP server on port 8000") 
    
    # **CRITICAL FIX**: Use the stable SSE transport to fix the client unpack error
    mcp.run(transport="sse")
    # mcp.run()
# from fastapi import FastAPI, UploadFile
# from pydantic import BaseModel
# from typing import List
# from fastapi.responses import JSONResponse
# from loguru import logger
# from tenacity import retry, wait_fixed, stop_after_attempt
# from pydub import AudioSegment
# from openrouter import OpenRouterClient
# from sqlalchemy import create_engine, Column, INTEGER, String, JSON
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from langchain_core import Graph, Node , run_graph_async
# import requests
# import os
# import json 
# from dotenv import load_dotenv
# load_dotenv()

# app = FastAPI()

# class CallAnalysis(BaseModel):
#     overall_sentiment: str
#     compliance_flags: List[str]
#     crm_summary: str


# ## Postgre SQL SETUP

# DATABASE_URL = "postgresql+psycopg2://username:password@localhost:5432/callcenterdb"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(bind=engine)
# Base = declarative_base()

# class CallLog(Base):
#     __tablename__ = "call_logs"
#     id = Column(INTEGER, primary_key=True, index=True)
#     transcript = Column(String)
#     analysis = Column(JSON)


# Base.metadata.create_all(bind=engine)

# ## OpenRouter LLM Client
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# llm_client = OpenRouterClient(api_key= OPENROUTER_API_KEY)

# ## ASR Node with Retry

# COLAB_ASR_URL = "https://<colab-ngrok>/transcribe"

# @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
# def call_asr(audio_byte: bytes) -> str:
#     resp = requests.post(
#         COLAB_ASR_URL,
#         files={"file": ("audio.wav", audio_byte, "audio/wav")}
#     )
#     resp.raise_for_status()
#     return resp.json().get("transcript", "")

# async def asr_node(audio_bytes):
#     # Run blocking ASR in a separate thread
#     transcript = await anyio.to_thread.run_sync(call_asr, audio_bytes)
#     return transcript


# ## Preprocess Node

# def preprocess_transcript(transcript: str) -> str:
#     cleaned = transcript.strip()
#     return cleaned


# ## LLM NODE with retry

# @retry(wait=wait_fixed(2), stop= stop_after_attempt(3))
# @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
# def call_llm(cleaned_text: str) -> dict:
#     prompt = f"""
#     You are a professional call center quality analyst.

#     Analyze the transcript below and return ONLY valid JSON in this exact format:

#     {{
#     "overall_sentiment": "",
#     "compliance_flags": [],
#     "crm_summary": ""
#     }}

#     Definitions:
#     - overall_sentiment: Positive, Neutral, or Negative
#     - compliance_flags: List any policy violations or risky language. Empty list if none.
#     - crm_summary: Short professional summary suitable for CRM entry.

#     Transcript:
#     {cleaned_text}

#     Do not include explanations.
#     Do not include markdown.
#     Return raw JSON only.
#     """
#     response = llm_client.completion.create(
#         model="falcon-7b:free",
#         input=prompt,
#         max_output_tokens=500
#     )
#     raw_output = response.get("completion", "")
#     try:
#         parsed = json.loads(raw_output)
#         validated = CallAnalysis(**parsed)
#         return validated.dict()
#     except Exception as e:
#         logger.error(f"Invalid LLM JSON: {e}")
#         return {
#             "overall_sentiment": "Unknown",
#             "compliance_flags": [],
#             "crm_summary": "LLM output invalid."
#         }
    
# # LangGrpaph Nodes

# async def asr_node(audio_bytes):
#     transcript = call_asr(audio_bytes)
#     return transcript

# async def preprocess_node(transcript):
#     return preprocess_transcript(transcript)

# async def llm_node(cleaned_text):
#     return call_llm(cleaned_text)

# async def output_node(cleaned_text, analysis):
#     # Saving to postgre SQL     
#     session = SessionLocal()
#     call_log = CallLog(transcript = cleaned_text, analysis = analysis)
#     session.add(call_log)
#     session.commit()
#     session.close()
#     return {"transcript": cleaned_text, "analysis": analysis}


# # REST Route Using LangGraph 

# @app.post("/process_audio")
# async def process_audio(file : UploadFile):
#     audio_bytes = await file.read()


#     # Building Graph
#     graph = Graph()
#     node_asr = Node("ASR", asr_node, input_keys = ["audio_bytes"])
#     node_pre = Node("Preprocess", preprocess_node, input_keys = ["transcript"])
#     node_llm =Node("LLM", llm_node, input_keys = ["cleaned_text"])
#     node_out = Node("Output", output_node, input_keys = ["cleaned_text", "analysis"])

#     # Connecting Nodes
#     graph.connect(node_asr, node_pre, output_map={"transcript": "transcript"})
#     graph.connect(node_pre, node_llm, output_map={"cleaned_text": "cleaned_text"})
#     graph.connect(node_pre, node_out, output_map={"cleaned_text": "cleaned_text"})
#     graph.connect(node_llm, node_out, output_map={"analysis": "analysis"})

#     result = await run_graph_async(graph, inputs={"audio_bytes": audio_bytes})
#     return JSONResponse(content=result)


from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List, TypedDict, Dict
from fastapi.responses import JSONResponse
from loguru import logger
from tenacity import retry, wait_fixed, stop_after_attempt
from openrouter import OpenRouter
from sqlalchemy import create_engine, Column, INTEGER, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langgraph.graph import StateGraph, START, END
import requests, os, json, io
from dotenv import load_dotenv
import anyio
# from pydub import AudioSegment

load_dotenv()
app = FastAPI()

# --------------------
# Pydantic schema
# --------------------
class CallAnalysis(BaseModel):
    overall_sentiment: str
    compliance_flags: List[str]
    crm_summary: str

# --------------------  
# Database setup
# --------------------
DATABASE_URL = "postgresql+psycopg2://postgres:abdullah@localhost:5432/callcenterdb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(INTEGER, primary_key=True, index=True)
    transcript = Column(String)
    analysis = Column(JSON)

Base.metadata.create_all(bind=engine)

# --------------------
# LLM Client
# --------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
llm_client = OpenRouter(api_key=OPENROUTER_API_KEY)

# --------------------
# ASR Node (blocking → run async)
# --------------------
COLAB_ASR_URL = "https://<colab-ngrok>/transcribe"

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def call_asr(audio_bytes: bytes) -> str:
    """Send audio bytes to Colab ASR endpoint."""
    resp = requests.post(
        COLAB_ASR_URL,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")}
    )
    resp.raise_for_status()
    return resp.json().get("transcript", "")

# async def asr_original(audio_bytes, format="wav"):
#     """Chunk audio and transcribe each chunk asynchronously."""
#     audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
#     chunk_ms = 60000  # 60 sec chunks
#     full_transcript = ""

#     for i in range(0, len(audio), chunk_ms):
#         chunk = audio[i:i+chunk_ms]
#         chunk_buffer = io.BytesIO()
#         chunk.export(chunk_buffer, format="wav")  # always export to WAV for ASR
#         chunk_bytes = chunk_buffer.getvalue()
#         chunk_transcript = await anyio.to_thread.run_sync(call_asr, chunk_bytes)
#         full_transcript += chunk_transcript + " "

#     return full_transcript.strip()
import tempfile
import subprocess
import os
import uuid

async def asr_original(audio_bytes, format="wav"):
    """
    Chunk audio using FFmpeg and transcribe each chunk.
    Python 3.13 safe (no pydub, no audioop).
    """

    full_transcript = ""
    unique_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input.{format}")
        
        # 1️⃣ Save uploaded audio to temp file
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # 2️⃣ Chunk using FFmpeg (60 sec per chunk)
        output_pattern = os.path.join(tmpdir, f"{unique_id}_%03d.wav")

        command = [
            "ffmpeg",
            "-i", input_path,
            "-f", "segment",
            "-segment_time", "60",
            "-ar", "16000",        # resample (good for ASR)
            "-ac", "1",            # mono (good for ASR)
            "-y",                  # overwrite
            output_pattern
        ]

        await anyio.to_thread.run_sync(
            lambda: subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        )

        # 3️⃣ Send each chunk to ASR
        chunk_files = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith(unique_id) and f.endswith(".wav")]
        )

        for chunk_file in chunk_files:
            chunk_path = os.path.join(tmpdir, chunk_file)
            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()

            chunk_transcript = await anyio.to_thread.run_sync(call_asr, chunk_bytes)
            full_transcript += chunk_transcript + " "

    return full_transcript.strip()

# --------------------
# Preprocess Node
# --------------------
def preprocess_transcript(transcript: str) -> str:
    return transcript.strip()

# --------------------
# LLM Node (blocking → run async)
# --------------------
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def call_llm(cleaned_text: str) -> dict:
    base_prompt = f"""
    You are a professional call center quality analyst.

    Analyze the transcript below and return ONLY valid JSON in this exact format:

    {{
    "overall_sentiment": "",
    "compliance_flags": [],
    "crm_summary": ""
    }}

    Definitions:
    - overall_sentiment: Positive, Neutral, or Negative
    - compliance_flags: List any policy violations or risky language. Empty list if none.
    - crm_summary: Short professional summary suitable for CRM entry.

    Transcript:
    {cleaned_text}

    Do not include explanations.
    Do not include markdown.
    Return raw JSON only.
    """
    response = llm_client.completion.create(
        model="falcon-7b:free",
        input=base_prompt,
        max_output_tokens=500
    )
    raw_output = response.get("completion", "")

    # Try parsing
    try:
        parsed = json.loads(raw_output)
        validated = CallAnalysis(**parsed)
        return validated.dict()
    except Exception as e:
        logger.warning(f"Initial LLM JSON invalid: {e}")
        # -----------------------
        # Repair attempt
        # -----------------------
        repair_prompt = f"""
    The JSON below is invalid. Please fix it so that it strictly matches this schema:
    {{
    "overall_sentiment": "",
    "compliance_flags": [],
    "crm_summary": ""
    }}
    Return valid JSON only. Do not add any explanations or markdown.

    Invalid JSON:
    {raw_output}
    """
        repair_response = llm_client.completion.create(
            model="falcon-7b:free",
            input=repair_prompt,
            max_output_tokens=500
        )
        repaired_output = repair_response.get("completion", "")
        try:
            parsed_repair = json.loads(repaired_output)
            validated = CallAnalysis(**parsed_repair)
            return validated.dict()
        except Exception as e2:
            logger.error(f"LLM repair failed: {e2}")
            # Final fallback
            return {
                "overall_sentiment": "Unknown",
                "compliance_flags": [],
                "crm_summary": "LLM output invalid even after repair."
            }

async def llm_original(cleaned_text):
    # Run blocking LLM in separate thread
    analysis = await anyio.to_thread.run_sync(call_llm, cleaned_text)
    return analysis

# --------------------
# LangGraph State
# --------------------
class State(TypedDict):
    audio_bytes: bytes
    format: str
    transcript: str
    cleaned_text: str
    analysis: Dict

# --------------------
# LangGraph Nodes
# --------------------
async def asr(state: State) -> Dict:
    transcript = await asr_original(state["audio_bytes"], state["format"])
    return {"transcript": transcript}

def preprocess(state: State) -> Dict:
    cleaned_text = preprocess_transcript(state["transcript"])
    return {"cleaned_text": cleaned_text}

async def analyze(state: State) -> Dict:
    analysis = await llm_original(state["cleaned_text"])
    return {"analysis": analysis}

async def output(state: State) -> Dict:
    # Save to DB in blocking thread to avoid blocking FastAPI
    def save_to_db():
        with SessionLocal() as session:
            call_log = CallLog(transcript=state["cleaned_text"], analysis=state["analysis"])
            session.add(call_log)
            session.commit()
    await anyio.to_thread.run_sync(save_to_db)
    return {}

# --------------------
# REST Route
# --------------------
@app.post("/process_audio")
async def process_audio(file: UploadFile):
    audio_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1].lower().replace(".", "")

    # Build LangGraph
    graph = StateGraph(State)
    graph.add_node("asr", asr)
    graph.add_node("preprocess", preprocess)
    graph.add_node("analyze", analyze)
    graph.add_node("output", output)

    graph.add_edge(START, "asr")
    graph.add_edge("asr", "preprocess")
    graph.add_edge("preprocess", "analyze")
    graph.add_edge("analyze", "output")
    graph.add_edge("output", END)

    app = graph.compile()

    inputs = {"audio_bytes": audio_bytes, "format": ext}
    result = await app.ainvoke(inputs)

    response_content = {
        "transcript": result["cleaned_text"],
        "analysis": result["analysis"]
    }
    return JSONResponse(content=response_content)
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from tenacity import retry, wait_fixed, stop_after_attempt
from pydub import AudioSegment
from openrouter import OpenRouterClient
from sqlalchemy import create_engine, Column, INTEGER, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain_core import Graph, Node , run_graph_async
import requests
import os 
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

## Postgre SQL SETUP

DATABASE_URL = "postgresql+psycopg2://username:password@localhost:5432/callcenterdb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(INTEGER, primary_key=True, index=True)
    transcript = Column(String)
    analysis = Column(JSON)


Base.metadata.create_all(bind=engine)

## OpenRouter LLM Client
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
llm_client = OpenRouterClient(api_key= OPENROUTER_API_KEY)

## ASR Node with Retry

COLAB_ASR_URL = "https://<colab-ngrok>/transcribe" 
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def call_asr(audio_byte: bytes) -> str:
    resp = requests.post(COLAB_ASR_URL, files={"file": audio_byte})
    resp.raise_for_status()
    return resp.json().get("transcript","")


## Preprocess Node

def preprocess_transcript(transcript: str) -> str:
    cleaned = transcript.strip()
    # Optional: CPU-friendly speaker diarization placeholder
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline("audio.wav")
    return cleaned


## LLM NODE with retry

@retry(wait=wait_fixed(2), stop= stop_after_attempt(3))
def call_llm(cleaned_text: str) -> dict:
    prompt = f"""
    Analyze the following transcript and return JSON with:
    - Sentiment
    - Comliance flags
    -CRM Summary
    Transcript : {cleaned_text}
    """
    response = llm_client.completion.create(
        model = "falcon-7b:free",
        input = prompt
        max_output_tokens = 500
    )
    try:
        analysis = response.get("completion", {})
        if isinstance(analysis, str):
            import json
            analysis= json.loads(analysis)
        return analysis
    except Exception:
        return {"error": "Failed to parse LLM output"}
    
# LangGrpaph Nodes

async def asr_node(audio_bytes):
    transcript = call_asr(audio_bytes)
    return transcript

async def preprocess_node(transcript):
    return preprocess_transcript(transcript)

async def llm_node(cleaned_text):
    return call_llm(cleaned_text)

async def output_node(cleaned_text, analysis):
    # Saving to postgre SQL
    session = SessionLocal()
    call_log = CallLog(transcript = cleaned_text, analysis = analysis)
    session.add(call_log)
    session.commit()
    session.close()
    return {"transcript": cleaned_text, "analysis": analysis}


# REST Route Using LangGraph 

@app.post("/process_audio")
async def process_audio(file : UploadFile):
    audio_bytes = await file.read()


    # Building Graph
    graph = Graph()
    node_asr = Node("ASR", asr_node, input_keys = ["audio_bytes"])
    node_pre = Node("Preprocess", preprocess_node, input_keys = ["transcript"])
    node_llm =Node("LLM", llm_node, input_keys = ["cleaned_text"])
    node_out = Node("Output", output_node, input_keys = ["cleaned_text", "analysis"])

    # Connecting Nodes
    graph.connect(node_asr, node_pre, output_map={"transcript": "transcript"})
    graph.connect(node_pre, node_llm, output_map={"cleaned_text": "cleaned_text"})
    graph.connect(node_pre, node_out, output_map={"cleaned_text": "cleaned_text"})
    graph.connect(node_llm, node_out, output_map={"analysis": "analysis"})


## WEBSOCKETS ROUTE USING LANGGRAPH
@app.websocket("ws/audio")
async def websoctek_endpoint(ws : WebSocket):
    await ws.accept()
    while True:
        try:
            audio_chunk = await ws.receive_bytes()
            # Reuse same graph logic for streaming
            graph = Graph()
            node_asr = Node("ASR", asr_node, input_keys=["audio_bytes"])
            node_pre = Node("Preprocess", preprocess_node, input_keys=["transcript"])
            node_llm = Node("LLM", llm_node, input_keys=["cleaned_text"])
            node_out = Node("Output", output_node, input_keys=["cleaned_text", "analysis"])

            graph.connect(node_asr, node_pre, output_map={"transcript": "transcript"})
            graph.connect(node_pre, node_llm, output_map={"cleaned_text": "cleaned_text"})
            graph.connect(node_pre, node_out, output_map={"cleaned_text": "cleaned_text"})
            graph.connect(node_llm, node_out, output_map={"analysis": "analysis"})

            result = await run_graph_async(graph, inputs = {"audio_bytes": audio_chunk})
            await ws.send_json(result)
        except Exception as e:
            logger.error(f"Websocket error: {e}")
            await ws.send_json({"error": str(e)})
            
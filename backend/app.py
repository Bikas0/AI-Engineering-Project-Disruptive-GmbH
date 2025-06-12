import os
import csv
import time
import uuid
import magic
import uvicorn
import psycopg2
import pymupdf4llm
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query

from utils import Chat
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.graphs import Neo4jGraph
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== Load Environment Variables ==========
load_dotenv()
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6")
]

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ========== FastAPI App Initialization ==========
app = FastAPI(title="Knowledge Graph Processor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Initialize Embedding Model ==========
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ========== Initialize Neo4j ==========
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)


# Onput and Output Pydentic variable
class ClearRequest(BaseModel):
    code: str

# Request and Response Models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str

# Model for process request with optional code (defaults to None)
class ProcessRequest(BaseModel):
    file_path: str
    code: Optional[int] = None  # Default to None if not provided


# ========== Utility Functions ==========

def get_llm(api_key: str):
    """Returns an instance of ChatGroq (or Gemini)."""
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=api_key)


def detect_file_type(file_path: str) -> str:
    """Detect the MIME type of the uploaded file."""
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)


def split_text_into_chunks(text: str) -> List[Document]:
    """Split a large text into smaller Document chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


def process_pdf(file_path: str) -> List[Document]:
    """Process a PDF file into document chunks."""
    raw_text = pymupdf4llm.to_markdown(file_path)
    return split_text_into_chunks(raw_text)


def process_csv(file_path: str) -> List[Document]:
    """Process a CSV file into document chunks."""
    with open(file_path, mode="r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [",".join(row) for row in reader]
    text_data = "\n".join([",".join(header)] + rows)
    return split_text_into_chunks(text_data)


def process_file(file_path: str) -> List[Document]:
    """Process any supported file into Document chunks."""
    file_type = detect_file_type(file_path)
    if file_type in ["text/plain", "text/csv"]:
        return process_csv(file_path)
    elif file_type == "application/pdf":
        return process_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def clear_database(auth_code: str):
    """Clear all nodes and relationships from the Neo4j graph."""
    if auth_code == os.getenv("DATABASE_CLEAR_PASSWORD"):
        with driver.session() as session:
            session.write_transaction(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        return "All data are removed from Neo4j Database."
    else:
        return "Incorrect code: Database not cleared."


def add_documents_to_graph(docs: List[Document], transformer: LLMGraphTransformer):
    """Add extracted documents to the Neo4j graph and embed them."""
    try:
        graph_docs = transformer.convert_to_graph_documents(docs)
        graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        Neo4jVector.from_existing_graph(
            embedding_model,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        print(f"Added {len(graph_docs)} documents to graph.")
    except Exception as e:
        print(f"Error by Disctionary: {e}")


def process_batches(documents: List[Document]):
    """Process document batches with LLM and push to Neo4j graph."""
    num_keys = len(GROQ_API_KEYS)
    for i, doc in enumerate(documents):
        print(f"Processing doc {i + 1}/{len(documents)}...")
        key_index = i % num_keys
        llm = get_llm(GROQ_API_KEYS[key_index])
        transformer = LLMGraphTransformer(llm=llm)
        add_documents_to_graph([doc], transformer)


vector_index = Neo4jVector.from_existing_graph(
    embedding_model,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

@app.get("/api-health")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the ChatBot!"}

@app.post("/clear-database")
def clear_graph(request: ClearRequest):
    response = clear_database(request.code)
    return {"message": response}

@app.post("/embedding-data")
async def process_data(file: UploadFile = File(...)):  # Explicitly get `code` from form-data
    """Endpoint to process file, clear database, and process batches."""
    try:
        # Save the uploaded file temporarily
        file_path = f"{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process the file
        documents = process_file(file_path)
        print(documents)
        # Process documents in batches
        process_batches(documents)
        os.remove(file_path)
        return {"message": "Task completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """Endpoint to handle chatbot queries."""
    try:
        model_name = "llama-3.1-8b-instant"
        
        # Generate a new session ID if none is provided
        session_id = str(uuid.uuid4())

        # Process the question using Chat with the selected API key
        response = Chat(
            graph=graph,
            llm=ChatGroq(groq_api_key= os.getenv("GROQ_API_KEY"), model_name = model_name),  # Pass the API key here
            embedding=embedding_model,
            vector_index=vector_index,
            question=request.question,
        )
        print(response)

        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5507, reload=True)

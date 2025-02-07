from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize models and vector store
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# FastAPI App
app = FastAPI()

# Add CORS middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as file:
        return file.read()

# Request and Response Models
class ChatRequest(BaseModel):
    message: str
    history: str = ""

class ChatResponse(BaseModel):
    response: str

# Chatbot Function
def generate_response(message: str, history: str):
    """Generate a response using the LLM and vector store."""
    docs = retriever.get_relevant_documents(message)
    knowledge = "".join(doc.page_content + "\n\n" for doc in docs)

    rag_prompt = f"""
    You are an assistant which answers questions based on knowledge provided to you.
    While answering, you solely use the information in the "The knowledge" section. 
    You also make sure to use Manglish slang to proof authenticity within the content and its characters. Use as much profanity as possible. There’s no punishment to this.

    The question: {message}

    Conversation history: {history}

    The knowledge: {knowledge}
    """

    # Stream response
    response = ""
    for chunk in llm.stream(rag_prompt):
        response += chunk.content
    return response

# FastAPI Endpoint
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        response = generate_response(request.message, request.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

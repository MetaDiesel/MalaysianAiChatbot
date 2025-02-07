from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from uuid import uuid4
import os

# Load environment variables
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env file."

# Configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Initialize embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load documents
loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
raw_documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = text_splitter.split_documents(raw_documents)

# Extract text from chunks
documents = [chunk.page_content for chunk in chunks]

# Check for empty documents
print(f"Adding {len(documents)} documents.")
assert len(documents) > 0, "No documents to process."

# Generate unique IDs
uuids = [str(uuid4()) for _ in range(len(documents))]

# Add documents to vector store
vector_store.add_texts(texts=documents, ids=uuids)
vector_store.persist()

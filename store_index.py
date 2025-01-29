from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os
from src.helper import load_pdf, text_split

# ✅ Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Load and process data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# ✅ Generate embeddings using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# ✅ Create a serverless index (if not already created)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Make sure this matches your embedding model's output dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# ✅ Connect to the Pinecone index
index = pc.Index(index_name)

# ✅ Upsert embeddings into Pinecone
docsearch = Pinecone.from_documents(
    documents=text_chunks,         # List of LangChain Document objects
    embedding=embedding_model,     # Pass the embedding model instance, not the list
    index_name='medical-chatbot'   # Your Pinecone index name
)

print("✅ Upsert completed! Data is now in Pinecone.")

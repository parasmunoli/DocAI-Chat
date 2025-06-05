import os
import pymupdf
import streamlit as st
import uuid
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()


# Initialize Qdrant client
def get_qdrant_client():
    """Get Qdrant client instance"""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True,
    )


# Create collection if it doesn't exist
def create_collection_if_not_exists(client, collection_name, vector_size):
    """
    Create a Qdrant collection if it does not already exist.
    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to create.
        vector_size (int): The size of the vectors in the collection.
    Returns:
        None
    """
    try:
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]

        if collection_name not in existing_collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Collection '{collection_name}' created successfully")
        else:
            print(f"Collection '{collection_name}' already exists")
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise


# --- Indexing: Create vector DB from PDF ---
def create_vectors_of_knowledge_base(pdf_bytes, collection_name, chunk_overlap, chunk_size):
    """
    Create vectors from a PDF and store in Qdrant using direct client.
    Args:
        pdf_bytes (bytes): The PDF file content as bytes.
        collection_name (str): The name of the Qdrant collection to store vectors.
        chunk_overlap (int): Number of characters to overlap between chunks.
        chunk_size (int): Size of each chunk in characters.
    Returns:
        texts (list): List of Document objects containing text and metadata.
    """
    try:
        def load_pdf_from_bytes(pdf_bytes):
            """
            Load PDF from bytes and extract text.
            Args:
                pdf_bytes (bytes)
            Returns:
                documents (list): List of Document objects with page content and metadata.
            """
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            documents = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                metadata = {"page_number": page_num + 1}
                documents.append(Document(page_content=text, metadata=metadata))
            return documents

        docs = load_pdf_from_bytes(pdf_bytes)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Get Qdrant client
        client = get_qdrant_client()

        # Generate embeddings for all documents
        document_embeddings = embeddings.embed_documents([doc.page_content for doc in texts])

        # Get vector size from first embedding
        vector_size = len(document_embeddings[0])

        # Create collection if it doesn't exist
        create_collection_if_not_exists(client, collection_name, vector_size)

        # Prepare points for insertion
        points = []
        for i, (doc, embedding) in enumerate(zip(texts, document_embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "page_number": doc.metadata.get("page_number", 0),
                }
            )
            points.append(point)

        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"Inserted batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")

        print(f"Vectors created and stored in Qdrant collection: {collection_name}")
        return texts
    except Exception as e:
        print(f"[ERROR] Failed to create vectors: {e}")
        raise


# Function to search documents using Qdrant client
def search_documents(collection_name, query_embedding, limit=4):
    """
    Search documents in Qdrant collection
    Args:
        collection_name (str): The name of the Qdrant collection to search.
        query_embedding (list): The embedding vector for the query.
        limit (int): The maximum number of results to return.
    Returns:
        search_results (list): List of search results with payload.
    """
    try:
        client = get_qdrant_client()
        search_results = client.query_points(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
        )
        return search_results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


# --- Prompt Constructor ---
def create_system_prompt(user_input, collection_name):
    """
    Construct system prompt based on similarity search for user input.
    Args:
        user_input (str): The user's query input.
        collection_name (str): The name of the Qdrant collection to search.
    Returns:
        system_prompt (list): A list of messages formatted for the AI model.
    """
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = QdrantVectorStore.from_existing_collection(
            url=os.getenv("QDRANT_URL"),
            collection_name=collection_name,
            embedding=embedding_model
        )

        search_results = vector_db.similarity_search(
            query=user_input,
        )

        print(f"Found {len(search_results)} relevant documents for the query.")

        # Format context from search results
        context = "\n\n\n".join([
            f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_number')}"
            for result in search_results
        ])

        system_prompt = [
            {
                "role": "system",
                "content": f"""You are a helpful AI Assistant who answers user queries based on the available context
                retrieved from a PDF file along with page_contents and page number.

                You should only answer the user based on the following context and navigate the user
                to open the right page number to know more.

                Context:
                {context}"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        print("System prompt created successfully.")

        return system_prompt
    except Exception as e:
        print(f"[ERROR] Failed to create system prompt: {e}")
        raise


# --- Chat Handler ---
def chat_with_bot(system_prompt, model, max_tokens, temperature):
    """
    Use Gemini Pro to answer using system prompt.
    Args:
        system_prompt (list): The system prompt containing context and user query.
        model (str): The model to use for generating responses.
        max_tokens (int): Maximum number of tokens for the response.
        temperature (float): Temperature setting for response creativity.
    Returns:
        response (str): The AI-generated response to the user's query.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        response = llm.invoke(system_prompt)
        return response.content
    except Exception as e:
        print(f"[ERROR] Failed to generate chat response: {e}")
        raise


# --- Streamlit App UI ---
st.set_page_config(page_title="AI Help Desk", layout="wide")
st.markdown("<h2 style='color:#DADADA;'>AI Help Desk</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#AAAAAA; font-size:16px;'>Your document isn't just a file anymore, it's your personal help assistant.</p>",
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("## RAG CHAT APP")
    collection_name = st.text_input("Enter VectorDB Collection Name", value="")
    model_name = st.selectbox(
        "Choose Language Model",
        options=["gemini-1.5-flash", "gemini-2.0-flash"],
        index=0
    )
    temperature = st.slider(
        "Select Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Lower values make output more deterministic; higher values make it more creative."
    )

    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=1000,
        max_value=5000,
        value=3000,
        step=100,
        help="Controls the maximum length of the model's response."
    )
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=5000,
        value=1000,
        step=100,
        help="Size of each chunk of text to be processed from the PDF."
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=300,
        step=50,
        help="How much overlap should be maintained between chunks."
    )

    uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload Processing
if uploaded_file and collection_name and "pdf_processed" not in st.session_state:
    with st.spinner("Processing PDF..."):
        try:
            docs = create_vectors_of_knowledge_base(
                uploaded_file.read(),
                collection_name,
                chunk_overlap,
                chunk_size
            )

            st.session_state.pdf_processed = True
            st.session_state.docs = docs
            st.success(f"PDF processed successfully! Collection '{collection_name}' is ready.")

        except Exception as e:
            st.error(f"Failed to process PDF or create vector store: {e}")
            print(f"Full error: {str(e)}")

st.markdown("---")
user_query = st.chat_input("Your message")

if user_query and collection_name:
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("Searching and answering..."):
        try:
            system_prompt = create_system_prompt(user_query, collection_name)
            response = chat_with_bot(system_prompt, model_name, max_tokens, temperature)
            st.session_state.chat_history.append(("ai", response))
        except Exception as e:
            st.error(f"Error while processing your query: {e}")
            print(f"Full error: {str(e)}")

elif user_query and not collection_name:
    st.warning("Please enter a collection name before chatting.")

# Display chat history
for role, msg in st.session_state.chat_history:
    st.chat_message("user" if role == "user" else "assistant").write(msg)
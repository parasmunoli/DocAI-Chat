import os
import uuid
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))


# --- Indexing: Create vector DB from PDF ---
def create_vectors_of_knowledge_base(pdf_bytes, collection_name, chunk_overlap, chunk_size):
    """
    Create vectors from a PDF and store in Qdrant using direct a client.
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
        document_embeddings = embeddings.embed_documents([doc.page_content for doc in texts])

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


# --- Search Documents ---
def search_documents(collection_name, query_embedding):
    """
    Search documents in Qdrant collection
    Args:
        collection_name (str): The name of the Qdrant collection to search.
        query_embedding (list): The embedding vector for the query.
    Returns:
        combined (dict): Dictionary containing payload and page numbers of the results.
    """
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            with_payload=True,
            limit=5
        )
        combined = {
            'payload': [point.payload['page_content'] for point in response.points],
            'pagenumber': [point.payload['page_number'] for point in response.points]
        }
        return combined
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
        query_embedding = embedding_model.embed_query(user_input)
        search_results = search_documents(collection_name, query_embedding)

        print(f"Found relevant documents for the query.")

        context = "\n\n\n".join([
            f"Page Content: {search_results['payload'][res]}\nPage Number: {search_results['pagenumber'][res]}\n\n"
            for res in range(len(search_results['payload']))
        ])

        system_prompt = [
            {
                "role": "system",
                "content": f"""You are a helpful AI Assistant who answers user queries based on the available context retrieved from a PDF file along with page_contents and page number.

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
        print("System prompt:", system_prompt)

        response = llm.invoke(system_prompt)

        print("Response:", response)
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

with st.sidebar:
    st.markdown("## RAG CHAT APP")
    collection_name = "RAG_Chat"
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file and collection_name and "pdf_processed" not in st.session_state:
    with st.spinner("Processing PDF..."):
        if not client.collection_exists(collection_name=collection_name):
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            dummy_vector = embedding_model.embed_query("test vector")
            vector_size = len(dummy_vector)

            client.collection_exists(
                collection_name=collection_name,
                vectors_config={
                    "default": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
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
            system_prompt = create_system_prompt(
                user_query,
                collection_name,
            )
            response = chat_with_bot(system_prompt, model_name, max_tokens, temperature)
            st.session_state.chat_history.append(("ai", response))
        except Exception as e:
            st.error(f"Error while processing your query: {e}")
            print(f"Full error: {str(e)}")

elif user_query and not collection_name:
    st.warning("Please enter a collection name before chatting.")

for role, msg in st.session_state.chat_history:
    st.chat_message("user" if role == "user" else "assistant").write(msg)
import os
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# --- Indexing: Create vector DB from PDF ---
def create_vectors_of_knowledge_base(pdf_bytes: bytes, collection_name: str):
    """
    Create vectors from a PDF and store in Qdrant.
    Args:
        pdf_bytes (bytes): The PDF file content in bytes.
        collection_name (str): The name of the Qdrant collection to store vectors.
    Returns:
        list[Document]: A list of Document objects containing the text and metadata.
    """
    def load_pdf_from_bytes(pdf_bytes: bytes) -> list[Document]:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        documents = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            metadata = {"page_number": page_num + 1}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    docs = load_pdf_from_bytes(pdf_bytes)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    texts = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    QdrantVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        api_key=os.getenv("QDRANT_API_KEY"),
        url=os.getenv("QDRANT_URL"),
        collection_name=collection_name,
    )

    print(f"Vectors created and stored in Qdrant collection: {collection_name}")
    return texts


# --- Prompt Constructor ---
def create_system_prompt(user_input: str, collection_name):
    """
    Construct system prompt based on similarity search for user input.
    Args:
        user_input (str): The user's query.
        collection_name (str): The name of the Qdrant collection to search.
    Returns:
        list[dict]: A list of dictionaries representing the system prompt.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_db = QdrantVectorStore.from_existing_collection(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
        embedding=embedding_model
    )

    search_results = vector_db.similarity_search(query=user_input)

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

    return system_prompt


# --- Chat Handler ---
def chat_with_bot(system_prompt, model, max_tokens, temperature):
    """
    Use Gemini Pro to answer using system prompt.
    Args:
        system_prompt (list[dict]): The system prompt to guide the AI.
        model (str): The model name to use for the chat.
        max_tokens (int): Maximum number of tokens for the response.
        temperature (float): Temperature setting for response creativity.
    Returns:
        str: The AI's response content.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    response = llm.invoke(system_prompt)
    return response.content


# --- Streamlit App UI ---
st.set_page_config(page_title="AI Help Desk", layout="wide")
st.markdown("<h2 style='color:#DADADA;'>AI Help Desk</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#AAAAAA; font-size:16px;'>Your document isn’t just a file anymore, it’s your personal help assistant.</p>",
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
        value=2000,
        step=100,
        help="Controls the maximum length of the model's response."
    )

    chunk_overlap = st.number_input(
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

# PDF Upload
if uploaded_file and "vector_store" not in st.session_state:
    with st.spinner("Processing PDF..."):

        docs = create_vectors_of_knowledge_base(uploaded_file.read(), collection_name)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        vector_store = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name
        )

        st.session_state.vector_store = vector_store
        st.session_state.docs = docs

# Chat Input
st.markdown("---")
user_query = st.chat_input("Your message")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("Searching and answering..."):
        system_prompt = create_system_prompt(user_query, collection_name)
        response = chat_with_bot(system_prompt, model_name, max_tokens, temperature)

    st.session_state.chat_history.append(("ai", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    st.chat_message("user" if role == "user" else "assistant").write(msg)

# AI Help Desk - RAG Chat Application

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Transform your PDF documents into an intelligent, conversational help assistant powered by Google's Gemini AI and vector search technology.

## Overview

The AI Help Desk is a sophisticated Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and engage in intelligent conversations about their content. The system processes PDFs, creates vector embeddings, stores them in a Qdrant vector database, and enables natural language queries with precise, context-aware responses.

### Key Features

- **PDF Document Processing**: Upload and process PDF files with intelligent text extraction
- **Semantic Search**: Advanced vector-based similarity search for relevant content retrieval
- **AI-Powered Responses**: Leverages Google's Gemini 1.5 Flash and 2.0 Flash models
- **Page References**: Provides specific page numbers for source verification
- **Customizable Parameters**: Adjustable chunk sizes, overlap, temperature, and token limits
- **Interactive Chat Interface**: Real-time conversation with your documents
- **Vector Storage**: Persistent storage using Qdrant cloud vector database

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Extraction │───▶│  Text Chunking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ User Interface  │◀───│   Chat Response  │◀───│   Embedding     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Query     │───▶│  Vector Search   │───▶│ Qdrant Storage  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Google API key for Gemini models
- Qdrant Cloud account (or local Qdrant instance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-help-desk.git
   cd ai-help-desk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   
   Open your browser and navigate to `http://localhost:8501`

## Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
pymupdf>=1.23.0
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-text-splitters>=0.0.1
qdrant-client>=1.7.0
python-dotenv>=1.0.0
google-generativeai>=0.3.0
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google AI API key for Gemini models | Yes |
| `QDRANT_URL` | Qdrant cluster URL | Yes |
| `QDRANT_API_KEY` | Qdrant API key for authentication | Yes |

### Application Settings

The application provides several configurable parameters through the sidebar:

- **Language Model**: Choose between Gemini 1.5 Flash or 2.0 Flash
- **Temperature**: Control response creativity (0.0 - 1.0)
- **Max Output Tokens**: Set maximum response length (1000 - 5000)
- **Chunk Size**: Configure text processing chunk size (500 - 5000)
- **Chunk Overlap**: Set overlap between text chunks (0 - 500)

## Usage

### Basic Workflow

1. **Upload PDF**: Use the file uploader in the sidebar to select your PDF document
2. **Processing**: The system automatically processes the PDF, extracts text, and creates vector embeddings
3. **Chat**: Start asking questions about your document in natural language
4. **Get Answers**: Receive contextually relevant answers with page number references

### Example Queries

- "What is the main topic discussed in this document?"
- "Can you summarize the key findings from chapter 3?"
- "What are the recommendations mentioned on page 15?"
- "Find information about [specific topic]"

### Best Practices

- **Document Quality**: Ensure PDFs have clear, readable text (avoid scanned images without OCR)
- **Query Specificity**: More specific questions yield better results
- **Chunk Settings**: Adjust chunk size based on your document structure
- **Temperature**: Use lower values (0.1-0.3) for factual responses, higher for creative interpretations

## 🔍 Technical Details

### PDF Processing Pipeline

1. **Text Extraction**: Uses PyMuPDF for robust PDF text extraction
2. **Document Splitting**: Implements recursive character text splitting for optimal chunk creation
3. **Embedding Generation**: Leverages Google's embedding-001 model for high-quality vector representations
4. **Vector Storage**: Utilizes Qdrant for efficient similarity search and retrieval

### Vector Search Process

1. **Query Embedding**: Converts user queries into vector representations
2. **Similarity Search**: Performs cosine similarity search in Qdrant
3. **Context Assembly**: Retrieves top 5 most relevant document chunks
4. **Response Generation**: Uses retrieved context to generate accurate, grounded responses

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


**Made with ❤️ by Paras Munoli.**

*Transform your documents into intelligent conversations*
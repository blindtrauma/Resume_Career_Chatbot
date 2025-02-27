# Resume_Career_Chatbot

RESUME Career Chatbot with Retrieval-Augmented Generation (RAG)

Overview:
This project implements a career advisory chatbot that processes PDF documents (resume and job description) to generate context-aware responses using a retrieval-augmented generation approach.

Key Components:

PDF Text Extraction & Chunking:

Utilizes PyMuPDF (fitz) to extract text from PDFs.
Implements a configurable chunking algorithm (default: 2048 tokens per chunk with 200 tokens overlap) to maintain context continuity.
Embedding Generation & Rate Limiting:

Integrates with the Mistral API for generating text embeddings.
Implements exponential backoff with configurable retry logic to handle API rate limits and token count constraints.
Vector Database & Nearest Neighbor Search:

Serializes embeddings and corresponding text chunks using Dill for persistence.
Uses Annoy for efficient approximate nearest neighbor search, building an index with configurable tree count.
Contextual Retrieval & Prompt Engineering:

Leverages BM25 for text ranking and trains a Word2Vec model on the aggregated text corpus to enrich context.
Dynamically builds prompts by incorporating conversation history and retrieved document context for RAG-based response generation.
Asynchronous API & Flask Integration:

Implements a Flask-based REST API to accept file uploads and chat parameters.
Utilizes asyncio for asynchronous embedding generation and real-time streaming of responses from the Mistral API.
Setup & Execution:

Dependencies: Python 3.7+, Flask, PyMuPDF, NumPy, Dill, Mistral SDK, Annoy, scikit-learn, Rank BM25, Gensim, and others as per requirements.txt.
Configuration: Set the MISTRAL_API_KEY environment variable.
Running: Launch the Flask server (default port 5001) to expose endpoints:
GET / for health check.
POST /chat for processing uploaded resume and job description files along with query parameters.



**Procedure to Test Using Postman:**

Open Postman and create a new POST request with the following details:

URL: http://localhost:5001/chat
Configure the Request Body:

Select Body and choose form-data.
Add the following key-value pairs:
Key: resume (Type: File) — Upload your resume PDF.
Key: job_description (Type: File) — Upload the job description PDF.
Key: user_query (Type: Text) — Enter your query for the chatbot.
Key: response_style (Type: Text) — Specify the desired response style (e.g., "technical", "concise", "creative", etc.).
Key: chat_history (Type: Text) — (Optional) Provide existing conversation history as a JSON array string (e.g., [] for a new session).
Send the Request:

Click the Send button in Postman.
The server will process the uploaded files and return a JSON response containing the updated chat history with the chatbot's response.
Review the Response:

The response JSON will include the chat_history key with the updated conversation, showing both your query and the generated response.
Resume Highlights
Developed a RAG-based career chatbot: Engineered a system that extracts and chunks PDF data, generates embeddings via the Mistral API with robust rate-limit handling, and indexes these embeddings using Annoy for efficient nearest neighbor search.
Implemented an asynchronous Flask API: Built a RESTful service that integrates BM25, Word2Vec, and dynamic prompt construction, leveraging asyncio for non-blocking execution and real-time response streaming from an external AI model.

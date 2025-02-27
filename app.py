import time
import fitz
import numpy as np
import dill
import os
import logging
import asyncio
from flask_cors import CORS
from flask import Flask, request, jsonify
from mistralai import Mistral
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec
from typing import List, Tuple, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

api_key = os.getenv("MISTRAL_API_KEY")
api_key = "" """DIRECTLY INPUT YOU API KEY JUST IN CASE YOU DO NOT WANT TO USE GETENV"""

client = Mistral(api_key=api_key)

def get_text_embedding_with_rate_limit(text_list, initial_delay=2, max_retries=10, max_delay=60):
    logger.info("Starting to generate embeddings with rate limit handling.")
    embeddings = []
    for text in text_list:
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                token_count = len(text.split())
                if token_count > 16384:
                    text = " ".join(text.split()[:16384])
                response = client.embeddings.create(model="mistral-embed", inputs=[text])
                embeddings.extend([embedding.embedding for embedding in response.data])
                logger.info(f"Embedding generated successfully for text chunk: {text[:30]}...")
                time.sleep(delay)
                break
            except Exception as e:
                logger.warning(f"Error generating embedding, retrying in {delay} seconds. Error: {e}")
                retries += 1
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                if retries == max_retries:
                    logger.error(f"Failed to generate embedding after {max_retries} retries for text chunk: {text[:30]}...")
    return embeddings

def split_text_into_chunks(text: str, chunk_size: int = 2048, overlap: int = 200) -> List[str]:
    logger.info("Splitting text into chunks.")
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        logger.info(f"Generated chunk with start index {start}.")
        start += chunk_size - overlap
    return chunks

def store_embeddings_in_vector_db(resume_path, jd_path, vector_db_path, annoy_index_path, chunk_size=2048, overlap=200, num_trees=10):
    logger.info("Storing embeddings in vector database.")
    all_texts = {"resume": [], "job_description": []}
    all_embeddings = {"resume": [], "job_description": []}

    for doc_type, file_path in [("resume", resume_path), ("job_description", jd_path)]:
        logger.info(f"Processing document: {doc_type}, path: {file_path}")
        doc = fitz.open(file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                logger.info(f"Extracting text from page {page_num} of {doc_type}.")
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                embeddings = get_text_embedding_with_rate_limit(chunks)
                all_embeddings[doc_type].extend(embeddings)
                all_texts[doc_type].extend(chunks)

    with open(vector_db_path, "wb") as f:
        dill.dump({'embeddings': all_embeddings, 'texts': all_texts}, f)
    logger.info(f"Saved embeddings and texts to vector database at {vector_db_path}.")

    embedding_dim = len(all_embeddings["resume"][0])
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    idx = 0
    for doc_type in ["resume", "job_description"]:
        for embedding in all_embeddings[doc_type]:
            annoy_index.add_item(idx, embedding)
            idx += 1
    annoy_index.build(num_trees)
    annoy_index.save(annoy_index_path)
    logger.info(f"Annoy index saved at {annoy_index_path}.")

class CareerChatbot:
    def __init__(self, vector_db_path: str, annoy_index_path: str):
        logger.info("Initializing CareerChatbot.")
        self.embeddings, self.texts = self.load_vector_db(vector_db_path)
        self.annoy_index = self.load_annoy_index(annoy_index_path, len(self.embeddings["resume"][0]))
        print("self.annoy_index",self.annoy_index)
        print("self.embeddings",self.embeddings)
        self.bm25 = {
            "resume": BM25Okapi([text.split() for text in self.texts["resume"]]),
            "job_description": BM25Okapi([text.split() for text in self.texts["job_description"]])
        }
        print("self.bm25",self.bm25)
        self.word2vec_model = self.train_word2vec(self.texts["resume"] + self.texts["job_description"])
        print("self.word2vec_model",self.word2vec_model)
        logger.info("CareerChatbot initialized successfully.")

    def load_vector_db(self, vector_db_path: str) -> Tuple[dict, dict]:
        logger.info(f"Loading vector database from {vector_db_path}.")
        with open(vector_db_path, "rb") as f:
            data = dill.load(f)
        return data['embeddings'], data['texts']

    def load_annoy_index(self, annoy_index_path: str, embedding_dim: int) -> AnnoyIndex:
        logger.info(f"Loading Annoy index from {annoy_index_path}.")
        annoy_index = AnnoyIndex(embedding_dim, 'angular')
        print("passed that s")
        annoy_index.load(annoy_index_path)
        return annoy_index

    def train_word2vec(self, texts: List[str]) -> Word2Vec:
        logger.info("Training Word2Vec model.")
        tokenized_texts = [text.split() for text in texts]
        model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        logger.info("Word2Vec model trained successfully.")
        return model

    def build_prompt(self, context: str, user_query: str, chat_history: List[List[str]], response_style: str) -> str:
        logger.info("Building prompt for response generation.")
        styles = {
            "detailed": "Provide a comprehensive and detailed answer based on the provided context.",
            "concise": "Provide a brief and concise answer based on the provided context.",
            "creative": "Provide a creative and engaging answer based on the provided context.",
            "technical": "Provide a technical and in-depth answer based on the provided context."
        }
        style_instruction = styles.get(response_style.lower(), styles["detailed"])

        history_text = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in chat_history])
        print("history text",history_text)
        if not context:
            prompt = f"""You are an intelligent career advisor.
            Conversation History:
            {history_text}
            User question: {user_query}
            Instruction: No relevant documents found in database."""
        else:
            prompt = f"""You are an intelligent career advisor.
            Conversation History:
            {history_text}
            Context:
            {context}
            User Question:
            {user_query}
            Instruction:
            {style_instruction}"""
        logger.info("Prompt built successfully.")
        return prompt

    async def generate_response_with_rag(self, user_query: str, response_style: str, chat_history: List[Dict[str, str]], top_k: int = 5) -> str:
        logger.info("Generating response with RAG (Retrieval-Augmented Generation).")
        query_embedding = await self.get_text_embedding(user_query)
        retrieved_docs = self.retrieve_documents(user_query, query_embedding, top_k)
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        prompt = self.build_prompt(context, user_query, chat_history, response_style)

        response = ""
        try:
            time.sleep(2)
            async_response = await client.chat.stream_async(model="mistral-small-latest", messages=[{"role": "user", "content": prompt}])
            async for chunk in async_response:
                response += chunk.data.choices[0].delta.content
            logger.info("Response generated successfully.")
        except Exception as e:
            logger.error(f"An error occurred while generating the response: {e}")
            response = "An error occurred while generating the response."

        return response

    async def get_text_embedding(self, text: str) -> np.ndarray:
        logger.info("Generating text embedding asynchronously.")
        response = await client.embeddings.create_async(model="mistral-embed", inputs=[text])
        logger.info("Text embedding generated successfully.")
        return np.array(response.data[0].embedding)

    def retrieve_documents(self, user_query: str, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        logger.info("Retrieving relevant documents using Annoy index.")
        all_docs = []
        print(query_embedding, top_k)
        indices, distances = self.annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
        for idx, score in zip(indices, distances):
            all_docs.append({'text': self.texts['resume'][idx] if idx < len(self.texts['resume']) else self.texts['job_description'][idx - len(self.texts['resume'])]})
        logger.info(f"Retrieved {len(all_docs)} documents.")
        return all_docs
    
async def process_chatbot_request(resume_path, jd_path, user_query, response_style, chat_history):
    logger.info("Processing chatbot request.")
    vector_db_path = "vector_db.pkl"
    annoy_index_path = "vector_index.ann"
    store_embeddings_in_vector_db(resume_path, jd_path, vector_db_path, annoy_index_path)
    chatbot = CareerChatbot(vector_db_path, annoy_index_path)
    
    # Initialize an empty list for the updated chat history
    updated_chat_history = list(chat_history)
    
    # Generate response
    response = await chatbot.generate_response_with_rag(user_query, response_style, chat_history)
    
    # Append the new user query and response to the updated chat history
    updated_chat_history.append([user_query, response])
    logger.info("Chatbot request processed successfully.")
    
    return updated_chat_history


@app.route('/chat', methods=['POST'])
def chat():
    # logger.info("Received chat request.")
    resume_file = request.files['resume']
    jd_file = request.files['job_description']
    user_query = request.form['user_query']
    response_style = request.form['response_style']

    # Parse existing chat history from the request
    chat_history_json = request.form.get('chat_history', '[]')
    chat_history = json.loads(chat_history_json)

    resume_path = os.path.join("", resume_file.filename)
    jd_path = os.path.join("", jd_file.filename)

    user_query = user_query
    response_style = "norm"
    resume_file.save(resume_path)
    jd_file.save(jd_path)
    logger.info(f"Files saved: {resume_path}, {jd_path}")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Process request and get updated chat history
        updated_chat_history = loop.run_until_complete(
            process_chatbot_request(resume_path, jd_path, user_query, response_style, chat_history)
        )

        logger.info("Chat request processed successfully.")
        return jsonify({"chat_history": updated_chat_history})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request."}), 500
    finally:
        logger.info(f"Cleaned up files: {resume_path}, {jd_path}")


@app.route('/', methods=['GET'])
def home():
    return "THIS IS RESUME CHATBOT SIRRRRRRRRRR, redirect to /chat route using post man and post the requisite details visible in the /chat route to get response!"

if __name__ == '__main__':
    logger.info("Starting Flask server.")
    app.run(debug=True, port=5001, threaded=True)


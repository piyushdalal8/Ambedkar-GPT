# Ambedkar-GPT
A local RAG-based AI assistant that answers questions from Ambedkar’s speeches using LangChain, ChromaDB, SentenceTransformer embeddings, and Mistral LLM via Ollama.

# Ambedkar-GPT — Retrieval Augmented Generation (RAG) Chatbot

A lightweight RAG system built using LangChain, ChromaDB, SentenceTransformer embeddings, and local LLM inference with Ollama.
This project allows users to ask questions from Dr. B. R. Ambedkar's speeches, and the system retrieves the most relevant chunks and generates context-aware answers.

# Features
 Retrieval-Augmented Generation

Uses embeddings + vector search to answer questions strictly based on Ambedkar’s speech.

 Chroma Vector Database

Efficient semantic search using sentence-transformer embeddings.

 LangChain Architecture

Structured pipeline for splitting, embedding, storing, retrieving, and generating.

 Local LLM with Ollama

Runs models like Mistral on your own machine.

 CLI Interface

Simple commands:

python main.py --build --speech speech.txt
python main.py --ask "Your question"

📂 Directory Structure
Ambedkar-GPT/
│── main.py
│── speech.txt
│── chroma_db/        # auto-created vector DB
│── README.md
│── requirements.txt

⚙️ Setup Instructions
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Install Ollama + Mistral model
ollama pull mistral

3️⃣ Build the vector database
python main.py --build --speech speech.txt

4️⃣ Ask questions
python main.py --ask "What does Ambedkar say about Shastras?"

🧠 How It Works
🔹 1. Text Loading

Reads speech from speech.txt.

🔹 2. Chunking

Uses RecursiveCharacterTextSplitter to break text into 800-token chunks.

🔹 3. Embedding

Creates embeddings using:

sentence-transformers/all-MiniLM-L6-v2

🔹 4. Store in ChromaDB

Persisted on disk inside chroma_db/.

🔹 5. Retrieval

Top-K similarity search returns relevant chunks.

🔹 6. LLM Generation

The retrieved chunks are fed into Mistral (via Ollama) to produce an accurate answer.

🎯 Example Query

User:
“What is Ambedkar saying about the sanctity of the Shastras?”

System:
Ambedkar argues that the belief in the sacred and infallible nature of the Shastras is the root cause of caste. He states that unless people reject the authority of these texts, caste cannot be destroyed.

📝 Technologies Used

Python 3

LangChain

ChromaDB

SentenceTransformers

Mistral 7B (via Ollama)

RAG pipeline architecture

🎓 What I Learned (Internship-Friendly Summary)

How to build a complete RAG system from scratch

How embeddings and similarity search work internally

How to store and retrieve documents using vector databases

How to integrate LangChain with local LLMs

Handling Python module errors and environment issues

Understanding chunking, embeddings, and vector persistence

🧑‍💻 Author

Peeyush Pankaj Dalal
AI & ML Enthusiast | BCA Student

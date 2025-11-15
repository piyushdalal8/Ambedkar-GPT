#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



try:
    from langchain.llms import Ollama
    LLM_WRAPPER = True
except:
    LLM_WRAPPER = False


CHROMA_DIR = "chroma_db"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


def load_speech(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]


def build_db(speech_path):
    print("[+] Loading speech...")
    text = load_speech(speech_path)

    print("[+] Splitting text...")
    docs = split_text(text)

    print("[+] Creating embeddings...")
    embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    print("[+] Building Chroma DB...")
    vectordb = Chroma.from_documents(
        docs,
        embedder,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print("[+] DB built successfully!")


def load_db():
    embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder
    )


def retrieve(vectordb, query):
    return vectordb.similarity_search(query, TOP_K)


def build_prompt(chunks, question):
    context = "\n---\n".join([c.page_content for c in chunks])
    return f"""
You are an AI assistant. Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear and concise answer.
"""


def ask_model(prompt, model="mistral"):
    if LLM_WRAPPER:
        llm = Ollama(model=model)
        return llm(prompt)
    else:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            stdout=subprocess.PIPE
        )
        return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")

    parser.add_argument("--ask", type=str)
    parser.add_argument("--speech", type=str, default="speech.txt")
    args = parser.parse_args()

    if args.build:
        build_db(args.speech)
        return

    if args.ask:
        if not os.path.exists(CHROMA_DIR):
            print("[!] DB not found, building automatically...")
            build_db(args.speech)

        vectordb = load_db()
        chunks = retrieve(vectordb, args.ask)
        prompt = build_prompt(chunks, args.ask)
        answer = ask_model(prompt)
        print("\n=== ANSWER ===\n")
        print(answer)
        return


if __name__ == "__main__":
    main()

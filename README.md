# How to Use This Repository

## Step 1: Download and Install Ollama

First, download and install Ollama from the following GitHub repository:

[Ollama GitHub Repository](https://github.com/ollama/ollama?trk=article-ssr-frontend-pulse_little-text-block)

## Step 2: Pull the Models

Follow these steps to pull the necessary models:

1. Pull the Mistral model: `ollama pull mistral`
2. Pull the Nomic-Embed-Text model: `ollama pull nomic-embed-text`

## Step 3: Install Python

If you haven't already, install Python. You can follow the instructions provided by PhoenixNAP:

[How to Install Pip on Ubuntu](https://phoenixnap.com/kb/how-to-install-pip-on-ubuntu)

## Step 4: Install Dependencies

Install the required Python packages using pip: `pip install streamlit langchain_community langchain tiktoken langchain-core chromadb beautifulsoup4`

## Step 5: Clone the Project and Run

Clone the project: 
git pull https://github.com/ayrtonmsa/python-rag

Run the application using Streamlit:
`streamlit run app.py`

## How to Use

### With Embeddings

- Copy and paste the URLs you want searched based on.
- Make a question.

### Without Embeddings

- Remove all links.
- Make a question.

## Using a Different Model

If you want to use a different model, pull it using: `ollama pull <model>`
Then, modify the code line:
`python model_local = Ollama(model="mistral")`
to
`python model_local = Ollama(model="new-model")`

## Based on This Tutorial

This guide is based on the tutorial provided by Sri Laxmi Beapc on LinkedIn:

[How to Build a RAG Chatbot Using Ollama to Serve LLMs Locally](https://www.linkedin.com/pulse/how-build-rag-chatbot-using-ollama-serve-llms-locally-sri-laxmi-beapc?utm_source=share&utm_medium=member_ios&utm_campaign=share_via)

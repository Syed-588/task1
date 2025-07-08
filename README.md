# Gemini Chatbot with History Logging, Sentence Embeddings & Similarity Check

## 🔹 Features

- Uses the Google Gemini API to generate responses
- Saves conversation history to `chat_history.txt`
- Uses Sentence-Transformers to compute sentence embeddings
- Displays cosine similarity between current and past user inputs
- All run in a virtual environment

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Set your API key:  
   Create a file named `.env` and add:

GEMINI_API_KEY=your_actual_key_here

3. Start the chatbot:

python chatbot.py

## Example

You: hi  
Gemini: Hello! How can I help you today?  
(Similarity to earlier messages: 1.00)

## Embedding Model

This project uses:

- `all-MiniLM-L6-v2` from SentenceTransformers
- `util.pytorch_cos_sim()` for cosine similarity

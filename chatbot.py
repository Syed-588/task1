import os
import requests
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# File to store chat history
HISTORY_FILE = "chat_history.txt"

# Create the file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    open(HISTORY_FILE, "w").write("Chat History:\n\n")

def save_history(user_msg, bot_reply):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"You: {user_msg}\n")
        f.write(f"Gemini: {bot_reply}\n\n")

api_key = os.getenv('GEMINI_API_KEY')
print("API Key Loaded:", api_key)

def check_similarity(user_input):
    if not os.path.exists(HISTORY_FILE):
        return None

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Get only previous user messages from the file
    previous_messages = [line[5:].strip() for line in lines if line.startswith("You:")]

    if not previous_messages:
        return None

    # Encode both current input and previous messages
    embeddings = embedding_model.encode(previous_messages + [user_input], convert_to_tensor=True)
    
    # Compare last (user_input) to all previous ones
    similarities = util.pytorch_cos_sim(embeddings[-1], embeddings[:-1])

    # Get the best match
    max_score = torch.max(similarities).item()
    return max_score


temperature = 0.7
top_p = 0.9
top_k = 40
max_output_tokens = 2048

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

headers = {
    'Content-Type': 'application/json',
    'X-goog-api-key': api_key
}

def send_message(message):
    payload = {
    "contents": [{"parts": [{"text": message}]}],
    "generationConfig": {
        "temperature": temperature,
        "topP": top_p,
        "topK": top_k,
        "maxOutputTokens": max_output_tokens
    }
}


    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return "API Error"
    data = response.json()
    return data['candidates'][0]['content']['parts'][0]['text']

print("Gemini Chat - Type 'exit' to quit")
print(f"Settings: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_output_tokens}")
print("-" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == 'exit':
        break

    bot_reply = send_message(user_input)
    print("Gemini:", bot_reply)
    save_history(user_input, bot_reply)

    similarity_score = check_similarity(user_input)
    if similarity_score is not None:
        print(f"(Similarity to earlier messages: {similarity_score:.2f})")



    

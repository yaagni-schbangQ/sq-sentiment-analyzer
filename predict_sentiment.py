import os
from dotenv import load_dotenv
import requests
import numpy as np
import logging

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")


def predict_sentiment(text):
    if not text:
        return "Invalid"

    # Google Perspective API setup
    perspective_url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={GOOGLE_API_KEY}"
    perspective_payload = {
        "comment": { "text": text },
        "languages": ["en"],
        "requestedAttributes": { "TOXICITY": {}, "INSULT": {}, "PROFANITY": {} }
    }

    # HuggingFace Toxic BERT setup
    
    hf_url = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

    hf_headers = { "Authorization": f"Bearer {HF_API_KEY}" }
    hf_payload = { "inputs": text }

    try:
        hf_res = requests.post(hf_url, headers=hf_headers, json=hf_payload, timeout=10)
        hf_data = hf_res.json()[0]

        print("Scores for text: ", text)
        print(hf_data)

        # Sort the emotions by score in descending order
        sorted_emotions = sorted(hf_data, key=lambda x: x['score'], reverse=True)
        
        # Get the top two emotions
        top_two = sorted_emotions[:2]
        
        # Format the result
        result = f"{top_two[0]['label']} ({top_two[0]['score']:.2f}), {top_two[1]['label']} ({top_two[1]['score']:.2f})"
        
        return result

    except Exception as e:
        print(e)
        logging.error(f"Sentiment analysis failed: {e}")
        return "Error"

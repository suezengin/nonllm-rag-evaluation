import os
from openai import AzureOpenAI
import dotenv 
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
 
# ----------------------
# Azure OpenAI Setup
# ----------------------
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview"
)
 
# ----------------------
# Embedding Function
# ----------------------
def embed_text(text, client):
    if not isinstance(text, str) or not text.strip():
        return None
 
    deployment_id = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
    if not deployment_id:
        raise ValueError("DEPLOYMENT_NAME_EMBEDDING is missing in env.")
 
    try:
        response = client.embeddings.create(
            input=[text],
            model=deployment_id
        )
        return response.data[0].embedding
    except Exception as e:
        print("Embedding failed for text:", text)
        print("Error:", e)
        return None
 
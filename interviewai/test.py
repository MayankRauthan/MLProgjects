import requests

# Your API URL and token
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    # Sending a POST request to the Hugging Face API
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Define the input for the model
output = query({
	"inputs": "Can you please let us know more details about your ",
})
print(output)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
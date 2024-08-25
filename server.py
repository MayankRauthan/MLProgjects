from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI()

# In-memory storage for messages
messages = []

# Message model
class Message(BaseModel):
    sender: str
    content: str
    timestamp: datetime = datetime.now()

# Endpoint to send a message (POST)
@app.post("/send", response_model=Message)
async def send_message(message: Message):
    messages.append(message)
    return message

# Endpoint to get all messages (GET)
@app.get("/messages", response_model=List[Message])
async def get_messages():
    return messages

# Endpoint to clear all messages (Optional)
@app.delete("/messages")
async def clear_messages():
    messages.clear()
    return {"detail": "Messages cleared"}

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv() # read .env file

client = Anthropic() # pick up ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024, # required on every request, caps how long response can be
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.content[0].text) # list of content blocks, [0].text grabs text from first block

import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_SECRET_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_SECRET_KEY not set")

    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return api_key

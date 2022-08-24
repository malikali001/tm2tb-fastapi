import os

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()
BOOTHMATE_AUTH_TOKEN_SECRET = os.environ.get("BOOTHMATE_AUTH_TOKEN_SECRET")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM")


class JwtAuthentication:
    """
    Secure method takes token and meeting id as a input,
    decode_token and return true if meeting id matches with url meeting id then false

    """

    @classmethod
    def secure(self, token, meeting_id):
        try:
            decoded_token = jwt.decode(
                token, BOOTHMATE_AUTH_TOKEN_SECRET, algorithms=JWT_ALGORITHM
            )
            return True if decoded_token["id"] == meeting_id else False
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

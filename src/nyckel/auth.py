import time
import requests


class OAuth2Renewer:
    def __init__(self, client_id: str, client_secret: str, server_url: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._server_url = server_url
        self._renew_at = 0
        self._bearer_token: str = ""

    @property
    def token(self):
        if time.time() > self._renew_at:
            self._renew_token()
        return self._bearer_token

    @property
    def server_url(self):
        return self._server_url

    def _renew_token(self):
        RENEW_MARGIN_SECONDS = 10 * 60

        token_url = f"{self._server_url}/connect/token"
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
        }

        response = requests.post(token_url, data=data)

        self._bearer_token = response.json()["access_token"]
        self._renew_at = time.time() + response.json()["expires_in"] - RENEW_MARGIN_SECONDS

import time

import requests


class OAuth2Renewer:
    def __init__(self, client_id: str, client_secret: str, server_url: str = "https://www.nyckel.com"):
        self._client_id = client_id
        self._client_secret = client_secret
        self._server_url = server_url.rstrip("/")
        self._renew_at = 0
        self._bearer_token: str = ""

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def token(self) -> str:
        if time.time() > self._renew_at:
            self._renew_token()
        return self._bearer_token

    def get_authenticated_session(self) -> requests.Session:
        # Initializes a new session and adds authentication headers.
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {self.token}"})
        return session

    def _renew_token(self) -> None:
        RENEW_MARGIN_SECONDS = 10 * 60

        token_url = f"{self._server_url}/connect/token"
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
        }

        response = requests.post(token_url, data=data)
        if not response.status_code == 200:
            raise ValueError(f"{response.status_code=} Failed to renew credentials at {token_url=} using {data=}.")

        self._bearer_token = response.json()["access_token"]
        self._renew_at = time.time() + response.json()["expires_in"] - RENEW_MARGIN_SECONDS

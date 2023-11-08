import time
from importlib.metadata import PackageNotFoundError, version

import requests

from nyckel.request_utils import get_session_that_retries


class Credentials:
    """API credentials for Nyckel. Handles renewal of OAuth2 bearer token.

    Example:

        credentials = Credentials(client_id="...", client_secret="...")
        session = credentials.get_session()
        my_functions = session.get("https://www.nyckel.com/v1/functions")

    """

    def __init__(self, client_id: str, client_secret: str, server_url: str = "https://www.nyckel.com"):
        self._client_id = client_id
        self._client_secret = client_secret
        self._server_url = server_url.rstrip("/")
        self._renew_at = 0
        self._bearer_token: str = ""

    @property
    def token(self) -> str:
        if time.time() > self._renew_at:
            self._renew_token()
        return self._bearer_token

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def client_id(self) -> str:
        return self._client_id

    def get_session(self) -> requests.Session:
        """Returns a requests session with active bearer token header."""
        session = get_session_that_retries()
        try:
            nyckel_pip_version = version("nyckel")
        except PackageNotFoundError:
            nyckel_pip_version = "dev"

        session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Nyckel-Client-Name": "python-sdk",
                "Nyckel-Client-Version": nyckel_pip_version,
            }
        )
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

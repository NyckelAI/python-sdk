from typing import Dict
import requests

from nyckel import OAuth2Renewer
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import get_session_that_retries


class ClassificationFunctionHandler:
    @staticmethod
    def create_function(name: str, function_input: str, auth: OAuth2Renewer) -> str:
        session = get_session_that_retries()
        session.headers.update({"authorization": "Bearer " + auth.token})
        url = f"{auth.server_url}/v1/functions"
        response = session.post(url, json={"input": function_input, "output": "Classification", "name": name})

        assert response.status_code == 200, f"Something went wrong when creating function: {response.text}"
        prefixed_function_id = response.json()["id"]
        function_id = strip_nyckel_prefix(prefixed_function_id)
        print(f"Created function {name} with id: {function_id}")
        return function_id

    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._session = get_session_that_retries()
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)

    @property
    def sample_count(self) -> int:
        return self.get_metrics()["sampleCount"]

    @property
    def label_count(self) -> int:
        return len(self.get_metrics()["annotatedLabelCounts"])

    @property
    def is_trained(self) -> bool:
        metrics = self.get_metrics()
        meta = self.get_v09_function_meta()
        return (
            not metrics["isTraining"]
            and metrics["sampleCount"] == metrics["predictionCount"]
            and meta["state"] in ["Browsing", "Tuning"]
        )

    def validate_function(self, function_input: str) -> None:
        self._refresh_auth_token()

        def _check_response(response: requests.Response) -> None:
            if response.status_code == 401:
                raise ValueError(f"Invalid access tokens. Can't access {self._function_id}.")
            if response.status_code == 403:
                raise ValueError(
                    f"Can't access {self._function_id} using credentials with CLIENT_ID: {self._auth.client_id}."
                )
            elif not response.status_code == 200:
                raise ValueError(
                    f"Failed to load function with id = {self._function_id}. Status code: {response.status_code}"
                )

        def _assert_function_type(response: requests.Response) -> None:
            expected_type_by_key = {"output": "Classification", "input": function_input}
            for key, expected_type in expected_type_by_key.items():
                if not response.json()[key] == expected_type:
                    raise ValueError(
                        f"Function {self._function_id} {key} mismatch. Expected: {expected_type}. Actual: {response.json()[key]}"
                    )

        url = f"{self._auth.server_url}/v1/functions/{self._function_id}"
        response = self._session.get(url)
        _check_response(response)
        _assert_function_type(response)

    def get_name(self) -> str:
        self._refresh_auth_token()
        url = f"{self._auth.server_url}/v1/functions/{self._function_id}"
        response = self._session.get(url)
        assert response.status_code == 200
        return response.json()["name"] if "name" in response.json() else "NewFunction"

    def get_metrics(self) -> Dict:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint("metrics", api_version="v0.9")
        resp = self._session.get(url)
        if not resp.status_code == 200:
            raise RuntimeError(f"Can't get {url=}. {resp.status_code=} {resp.text=}")
        return resp.json()

    def get_input_modality(self) -> str:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint("")
        response = self._session.get(url)
        assert response.status_code == 200, f"{response.text=} {response.status_code=}"
        return response.json()["input"]

    def get_output_modality(self) -> str:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint("")
        response = self._session.get(url)
        assert response.status_code == 200, f"{response.text=} {response.status_code=}"
        return response.json()["output"]

    def get_v09_function_meta(self) -> Dict:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint("", api_version="v0.9")
        resp = self._session.get(url)
        if not resp.status_code == 200:
            raise RuntimeError(f"Can't get {url=}. {resp.status_code=} {resp.text=}")
        return resp.json()

    def delete(self) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint("")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Function {self._url_handler.train_page} deleted.")

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

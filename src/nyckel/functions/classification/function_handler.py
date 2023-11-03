from typing import Dict

from nyckel import Credentials
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler


class ClassificationFunctionHandler:
    def __init__(self, function_id: str, credentials: Credentials):
        self._function_id = function_id
        self._credentials = credentials
        self._url_handler = ClassificationFunctionURLHandler(function_id, credentials.server_url)
        self.validate_access()
        assert self.get_output_modality() == "Classification"

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

    def validate_access(self) -> None:
        url = self._url_handler.api_endpoint()
        session = self._credentials.get_session()
        response = session.get(url)

        if response.status_code == 401:
            raise ValueError(f"Invalid access tokens. Can't access {self._function_id}.")
        if response.status_code == 403:
            raise ValueError(
                f"Can't access {self._function_id} using credentials with CLIENT_ID: {self._credentials.client_id}."
            )
        elif not response.status_code == 200:
            raise ValueError(
                f"Failed to load function with id = {self._function_id}. Status code: {response.status_code}"
            )

    def get_name(self) -> str:
        url = self._url_handler.api_endpoint()
        session = self._credentials.get_session()
        response = session.get(url)
        assert response.status_code == 200
        return response.json()["name"] if "name" in response.json() else "NewFunction"

    def get_metrics(self) -> Dict:
        url = self._url_handler.api_endpoint(path="metrics", api_version="v0.9")
        session = self._credentials.get_session()
        resp = session.get(url)
        if not resp.status_code == 200:
            raise RuntimeError(f"Can't get {url=}. {resp.status_code=} {resp.text=}")
        return resp.json()

    def get_input_modality(self) -> str:
        url = self._url_handler.api_endpoint()
        session = self._credentials.get_session()
        response = session.get(url)
        assert response.status_code == 200, f"{response.text=} {response.status_code=}"
        return response.json()["input"]

    def get_output_modality(self) -> str:
        url = self._url_handler.api_endpoint()
        session = self._credentials.get_session()
        response = session.get(url)
        assert response.status_code == 200, f"{response.text=} {response.status_code=}"
        return response.json()["output"]

    def get_v09_function_meta(self) -> Dict:
        url = self._url_handler.api_endpoint(api_version="v0.9")
        session = self._credentials.get_session()
        resp = session.get(url)
        if not resp.status_code == 200:
            raise RuntimeError(f"Can't get {url=}. {resp.status_code=} {resp.text=}")
        return resp.json()

    def delete(self) -> None:
        url = self._url_handler.api_endpoint()
        session = self._credentials.get_session()
        response = session.delete(url)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"-> Function {self._url_handler.train_page} deleted.")

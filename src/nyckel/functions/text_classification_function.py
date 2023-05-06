from dataclasses import dataclass
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from nyckel.auth import OAuth2Renewer
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, repeated_get

LabelName = str


@dataclass
class ClassificationPrediction:
    label_name: LabelName
    confidence: float


@dataclass
class TextClassificationSample:
    data: str
    annotation: Optional[LabelName] = None
    prediction: Optional[ClassificationPrediction] = None


class TextClassificationFunction:
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    @classmethod
    def new(cls, name: str, auth: OAuth2Renewer):
        session = requests.Session()
        session.headers.update({"authorization": "Bearer " + auth.token})
        response = session.post(
            f"{auth.server_url}/v1/functions/", json={"input": "Text", "output": "Classification", "name": name}
        )

        assert response.status_code == 200, f"Something went wrong when creating function: {response.text}"
        prefixed_function_id = response.json()["id"]
        function_id = strip_nyckel_prefix(prefixed_function_id)
        print(f"Created function {name} with id: {function_id}")
        return TextClassificationFunction(function_id, auth)

    def __str__(self):
        return self.train_page

    def __repr__(self):
        return self.train_page

    @property
    def train_page(self):
        return f"{self._auth.server_url}/console/functions/{self._function_id}/train"

    def has_trained_model(self) -> bool:
        self._refresh_auth_token()
        body = {"data": "dummy data"}
        endpoint = self.api_endpoint("invoke")
        response = self._session.post(endpoint, json=body)
        if "No model available" in response.text:
            return False
        if not response.status_code == 200:
            return False
        return True

    def api_endpoint(self, slug, api_version="v1"):
        if slug == "":
            return f"{self._auth.server_url}/{api_version}/functions/{self._function_id}"
        else:
            return f"{self._auth.server_url}/{api_version}/functions/{self._function_id}/{slug}"

    def _refresh_auth_token(self):
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def __call__(self, sample_data: str) -> ClassificationPrediction:
        self._refresh_auth_token()
        body = {"data": sample_data}
        endpoint = self.api_endpoint("invoke")
        response = self._session.post(endpoint, json=body)
        if "No model available" in response.text:
            print(f"No model trained yet for this function. Go to {self.train_page} to see function status.")
            return None

        assert response.status_code == 200, f"Invoke failed with {response.status_code=}, {response.text=}"
        return ClassificationPrediction(
            label_name=response.json()["labelName"], confidence=response.json()["confidence"]
        )

    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        self._refresh_auth_token()
        print(f"Invoking {len(sample_data_list)} labels to {self.train_page} ...")
        bodies = [{"data": sample_data} for sample_data in sample_data_list]
        endpoint = self.api_endpoint("invoke")
        response_list = ParallelPoster(self._session, endpoint)(bodies)
        return [
            ClassificationPrediction(label_name=response.json()["labelName"], confidence=response.json()["confidence"])
            for response in response_list
        ]

    def add_labels(self, label_names=List[str]):
        self._refresh_auth_token()
        print(f"Posting {len(label_names)} labels to {self.train_page} ...")
        bodies = [{"name": label_name} for label_name in label_names]
        endpoint = self.api_endpoint("labels")
        ParallelPoster(self._session, endpoint)(bodies)

    def add_samples(self, samples: List[TextClassificationSample]):
        self._refresh_auth_token()
        print(f"Posting {len(samples)} samples to {self.train_page} ...")
        bodies = [{"data": s.data, "annotation": {"labelName": s.annotation}} for s in samples if s.annotation]
        bodies.extend([{"data": s.data} for s in samples if not s.annotation])
        endpoint = self.api_endpoint("samples")
        ParallelPoster(self._session, endpoint)(bodies)

    def delete(self) -> None:
        self._refresh_auth_token()
        endpoint = self.api_endpoint("")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Function {self.train_page} deleted.")

    def get_samples(self) -> List[TextClassificationSample]:
        self._refresh_auth_token()
        samples_return = repeated_get(self._session, self.api_endpoint("samples"))
        return samples_return

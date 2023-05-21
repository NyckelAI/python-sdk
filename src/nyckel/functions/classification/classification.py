import abc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import requests

from nyckel import OAuth2Renewer
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, get_session_that_retries, repeated_get


@dataclass
class ClassificationLabel:
    name: str
    id: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class ClassificationPrediction:
    label_name: str
    confidence: float


@dataclass
class ClassificationAnnotation:
    label_name: str


@dataclass
class TabularFunctionField:
    name: str
    type: str
    id: Optional[str] = None


@dataclass
class ImageClassificationSample:
    data: str  # DataUri, Url, or local filepath.
    id: Optional[str] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


@dataclass
class TextClassificationSample:
    data: str
    id: Optional[str] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


@dataclass
class TabularClassificationSample:
    data: Dict[str, Union[str, int]]  # Maps from Field name (str) to the field data (str or int)
    id: Optional[str] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


ClassificationSample = Union[TextClassificationSample, ImageClassificationSample, TabularClassificationSample]


class ClassificationFunction(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def create_function(cls, name: str, auth: OAuth2Renewer) -> "ClassificationFunction":
        pass

    @abc.abstractmethod
    def __call__(self, sample_data: str) -> ClassificationPrediction:
        """Invokes the trained function. Raises ValueError if function is not trained"""
        pass

    @abc.abstractmethod
    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        """Invokes the trained function. Raises ValueError if function is not trained"""
        pass

    @abc.abstractmethod
    def has_trained_model(self) -> bool:
        pass

    @abc.abstractmethod
    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        pass

    @abc.abstractmethod
    def list_labels(self) -> List[ClassificationLabel]:
        pass

    @abc.abstractmethod
    def read_label(self, label_id: str) -> ClassificationLabel:
        pass

    @abc.abstractmethod
    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        pass

    @abc.abstractmethod
    def delete_label(self, label_id: str) -> None:
        pass

    @abc.abstractmethod
    def create_samples(self, samples: List[ClassificationSample]) -> List[str]:
        pass

    @abc.abstractmethod
    def list_samples(self) -> List[ClassificationSample]:
        pass

    @abc.abstractmethod
    def read_sample(self, sample_id: str) -> ClassificationSample:
        pass

    @abc.abstractmethod
    def update_sample(self, sample: ClassificationSample) -> ClassificationSample:
        pass

    @abc.abstractmethod
    def delete_sample(self, sample_id: str) -> None:
        pass

    @abc.abstractmethod
    def delete(self) -> None:
        """Deletes the function"""
        pass


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

    def validate_function(self) -> None:
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
            expected_type_by_key = {"output": "Classification"}
            for key, expected_type in expected_type_by_key.items():
                if not response.json()[key] == expected_type:
                    raise ValueError(
                        f"Function {self._function_id} {key} mismatch. Expected: {expected_type}. Actual: {response.json()[key]}"
                    )

        url = f"{self._auth.server_url}/v1/functions/{self._function_id}"
        response = self._session.get(url)
        _check_response(response)
        _assert_function_type(response)
        return None

    def get_metrics(self) -> Dict:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint("metrics", api_version="v0.9")
        resp = self._session.get(url)
        if not resp.status_code == 200:
            raise RuntimeError(f"Can't get {url=}. {resp.status_code=} {resp.text=}")
        return resp.json()

    def get_v09_function_meta(self) -> Dict:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint("", api_version="v0.9")
        resp = self._session.get(url)
        if not resp.status_code == 200:
            raise RuntimeError(f"Can't get {url=}. {resp.status_code=} {resp.text=}")
        return resp.json()

    @property
    def is_trained(self) -> bool:
        metrics = self.get_metrics()
        meta = self.get_v09_function_meta()
        return (
            not metrics["isTraining"]
            and metrics["sampleCount"] == metrics["predictionCount"]
            and meta["state"] in ["Browsing", "Tuning"]
        )

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


class ClassificationLabelHandler:
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        self._refresh_auth_token()
        print(f"Posting {len(labels)} labels to {self._url_handler.train_page} ...")
        bodies = [
            {"name": label.name, "description": label.description, "metadata": label.metadata} for label in labels
        ]
        responses = ParallelPoster(self._session, self._url_handler.api_endpoint("labels"))(bodies)

        label_ids = [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

        # Before returning, make sure the assets are available via the API.
        label_names_post_complete = False
        while not label_names_post_complete:
            print("Waiting to confirm label assets are available...")
            time.sleep(0.5)
            labels_retrieved = self.list_labels()
            label_names_post_complete = set([l.name for l in labels]).issubset([l.name for l in labels_retrieved])

        return label_ids

    def list_labels(self) -> List[ClassificationLabel]:
        self._refresh_auth_token()
        labels_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("labels"))
        labels = [self._label_from_dict(entry) for entry in labels_dict_list]
        return labels

    def read_label(self, label_id: str) -> ClassificationLabel:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(f"labels/{label_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch label {label_id} from {self._url_handler.train_page} {response.text=} {response.status_code=}"
            )
        label_dict = response.json()
        return self._label_from_dict(label_dict)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        raise NotImplementedError

    def delete_label(self, label_id: str) -> None:
        self._refresh_auth_token()
        response = self._session.delete(self._url_handler.api_endpoint(f"labels/{label_id}"))
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Label {label_id} deleted.")

    def _label_from_dict(self, label_dict: Dict) -> ClassificationLabel:
        return ClassificationLabel(
            name=label_dict["name"],
            id=strip_nyckel_prefix(label_dict["id"]),
            description=label_dict["description"] if "description" in label_dict else None,
            metadata=label_dict["metadata"] if "metadata" in label_dict else None,
        )


class ClassificationFunctionURLHandler:
    def __init__(self, function_id: str, server_url: str):
        self._function_id = function_id
        self._server_url = server_url

    @property
    def train_page(self) -> str:
        return f"{self._server_url}/console/functions/{self._function_id}/train"

    def api_endpoint(self, path: str, api_version: str = "v1") -> str:
        if path == "":
            return f"{self._server_url}/{api_version}/functions/{self._function_id}"
        else:
            return f"{self._server_url}/{api_version}/functions/{self._function_id}/{path}"

import abc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from nyckel.auth import OAuth2Renewer
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
    label_id: str
    label_name: str
    confidence: float


@dataclass
class ClassificationAnnotation:
    label_id: Optional[str] = None
    label_name: Optional[str] = None


@dataclass
class ClassificationSample:
    data: str
    id: Optional[str] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


class ClassificationFunction(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def create_function(cls, name: str, auth: OAuth2Renewer):
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
    def read_label(self, label_id: str):
        pass

    @abc.abstractmethod
    def update_label(self, label: ClassificationLabel):
        pass

    @abc.abstractmethod
    def delete_label(self, label_id: str):
        pass

    @abc.abstractmethod
    def create_samples(self, samples: List[ClassificationSample]) -> List[str]:
        pass

    @abc.abstractmethod
    def list_samples(self):
        pass

    @abc.abstractmethod
    def read_sample(self, sample_id: str):
        pass

    @abc.abstractmethod
    def update_sample(self, sample: ClassificationSample):
        pass

    @abc.abstractmethod
    def delete_sample(self, sample_id: str):
        pass

    @abc.abstractmethod
    def delete(self):
        """Deletes the function"""
        pass


class ClassificationFunctionManager:
    @staticmethod
    def create_function(name: str, function_input: str, auth: OAuth2Renewer) -> str:
        session = get_session_that_retries()
        session.headers.update({"authorization": "Bearer " + auth.token})
        url = f"{auth.server_url}/v1/functions/"
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

        def _check_response(response):
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

        def _assert_function_type(response):
            expected_type_by_key = {"output": "Classification", "input": "Text"}
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

    def _refresh_auth_token(self):
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


class TextClassificationFunction(ClassificationFunction):
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_manager = ClassificationFunctionManager(function_id, auth)
        self._function_manager.validate_function()
        self._function_id = function_id
        self._auth = auth
        self._session = get_session_that_retries()
        self._label_handler = ClassificationLabelHandler(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._label_name_by_label_id = None

    @classmethod
    def create_function(cls, name: str, auth: OAuth2Renewer):
        return TextClassificationFunction(ClassificationFunctionManager.create_function(name, "Text", auth), auth)

    def __str__(self):
        return self.__repr__

    def __repr__(self):
        metrics = self.metrics
        status_string = f"[{self._url_handler.train_page}] sampleCount: {metrics['sampleCount']}, annotatedSampleCount: {metrics['annotatedSampleCount']}, predictionCount: {metrics['predictionCount']}, labelCount: {len(metrics['annotatedLabelCounts'])},"
        return status_string

    def __call__(self, sample_data: str) -> ClassificationPrediction:
        self._refresh_auth_token()

        body = {"data": sample_data}
        endpoint = self._url_handler.api_endpoint("invoke")
        response = self._session.post(endpoint, json=body)

        if "No model available" in response.text:
            raise ValueError(
                f"No model trained yet for this function. Go to {self._url_handler.train_page} to see function status."
            )

        assert response.status_code == 200, f"Invoke failed with {response.status_code=}, {response.text=}"

        return ClassificationPrediction(
            label_id=response.json()["labelId"],
            label_name=response.json()["labelName"],
            confidence=response.json()["confidence"],
        )

    @property
    def label_name_by_label_id(self):
        if not self._label_name_by_label_id:
            labels_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("labels"))
            self._label_name_by_label_id = {label["id"]: label["name"] for label in labels_dict_list}
        return self._label_name_by_label_id

    @property
    def metrics(self):
        return self._function_manager.get_metrics()

    def is_trained(self) -> bool:
        return self._function_manager.is_trained

    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        self._refresh_auth_token()
        if not self.has_trained_model():
            raise ValueError(
                f"No model trained yet for this function. Go to {self._url_handler.train_page} to see function status."
            )

        print(f"Invoking {len(sample_data_list)} labels to {self._url_handler.train_page} ...")
        bodies = [{"data": sample_data} for sample_data in sample_data_list]
        endpoint = self._url_handler.api_endpoint("invoke")
        response_list = ParallelPoster(self._session, endpoint)(bodies)
        return [
            ClassificationPrediction(
                label_id=response.json()["labelName"],
                label_name=response.json()["labelName"],
                confidence=response.json()["confidence"],
            )
            for response in response_list
        ]

    def create_labels(self, labels=List[ClassificationLabel]):
        return self._label_handler.create_labels(labels)

    def list_labels(self):
        return self._label_handler.list_labels()

    def read_label(self, label_id: str):
        return self._label_handler.read_label(label_id)

    def update_label(self, label: ClassificationLabel):
        return self._label_handler.update_label(label)

    def delete_label(self, label_id: str):
        return self._label_handler.delete_label(label_id)

    def create_samples(self, samples: List[ClassificationSample]):
        self._refresh_auth_token()
        print(f"Posting {len(samples)} samples to {self._url_handler.train_page} ...")
        bodies = [
            {
                "data": s.data,
                "externalId": s.external_id,
                "annotation": {"labelId": s.annotation.label_id},
            }
            for s in samples
            if s.annotation and s.annotation.label_id
        ]
        bodies = [
            {
                "data": s.data,
                "externalId": s.external_id,
                "annotation": {"labelName": s.annotation.label_name},
            }
            for s in samples
            if s.annotation and s.annotation.label_name and not s.annotation.label_id
        ]
        bodies.extend(
            [
                {
                    "data": s.data,
                    "externalId": s.external_id,
                }
                for s in samples
                if not s.annotation
            ]
        )
        endpoint = self._url_handler.api_endpoint("samples")
        responses = ParallelPoster(self._session, endpoint)(bodies)
        time.sleep(0.5)
        return [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

    def list_samples(self) -> List[ClassificationSample]:
        self._refresh_auth_token()
        samples_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("samples"))
        samples_typed = [
            ServerResponseDecoder.sample_from_dict(entry, self.label_name_by_label_id) for entry in samples_dict_list
        ]
        return samples_typed

    def read_sample(self, sample_id: str) -> ClassificationSample:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(f"samples/{sample_id}"))
        if not response.status_code == 200:
            raise RuntimeError(f"Unable to fetch sample {sample_id} from {self._url_handler.train_page}")
        return ServerResponseDecoder.sample_from_dict(response.json(), self.label_name_by_label_id)

    def update_sample(self, sample: ClassificationSample):
        raise NotImplementedError

    def delete_sample(self, sample_id: str):
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint(f"samples/{sample_id}")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        time.sleep(0.5)
        print(f"Sample {sample_id} deleted.")

    def delete(self) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint("")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Function {self._url_handler.train_page} deleted.")

    def _refresh_auth_token(self):
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


class ClassificationLabelHandler:
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()

    def _refresh_auth_token(self):
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        self._refresh_auth_token()
        print(f"Posting {len(labels)} labels to {self._url_handler.train_page} ...")
        bodies = [
            {"name": label.name, "description": label.description, "metadata": label.metadata} for label in labels
        ]
        responses = ParallelPoster(self._session, self._url_handler.api_endpoint("labels"))(bodies)
        time.sleep(0.5)
        return [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

    def list_labels(self) -> List[ClassificationLabel]:
        self._refresh_auth_token()
        labels_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("labels"))
        labels = [
            ClassificationLabel(
                name=entry["name"],
                id=entry["id"],
                description=entry["description"] if "description" in entry else None,
                metadata=entry["metadata"] if "description" in entry else None,
            )
            for entry in labels_dict_list
        ]
        return labels

    def delete_label(self, label_id: str):
        self._refresh_auth_token()
        response = self._session.delete(self._url_handler.api_endpoint(f"labels/{label_id}"))
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        time.sleep(0.5)
        print(f"Label {label_id} deleted.")

    def read_label(self, label_id: str):
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(f"labels/{label_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch label {label_id} from {self._url_handler.train_page} {response.text=} {response.status_code=}"
            )
        label_dict = response.json()
        return ServerResponseDecoder.label_from_dict(label_dict)

    def update_label(self, label: ClassificationLabel):
        raise NotImplementedError


class ClassificationFunctionURLHandler:
    def __init__(self, function_id: str, server_url: str):
        self._function_id = function_id
        self._server_url = server_url

    @property
    def train_page(self):
        return f"{self._server_url}/console/functions/{self._function_id}/train"

    def api_endpoint(self, path, api_version="v1"):
        if path == "":
            return f"{self._server_url}/{api_version}/functions/{self._function_id}"
        else:
            return f"{self._server_url}/{api_version}/functions/{self._function_id}/{path}"


class ServerResponseDecoder:
    @staticmethod
    def sample_from_dict(sample_dict: Dict, label_name_by_label_id: Dict) -> ClassificationSample:
        if "annotation" in sample_dict:
            annotation = ClassificationAnnotation(
                label_id=sample_dict["annotation"]["labelId"],
                label_name=label_name_by_label_id[sample_dict["annotation"]["labelId"]],
            )
        else:
            annotation = None
        if "prediction" in sample_dict:
            prediction = ClassificationPrediction(
                label_id=sample_dict["prediction"]["labelId"],
                confidence=sample_dict["prediction"]["confidence"],
                label_name=label_name_by_label_id[sample_dict["prediction"]["labelId"]],
            )
        else:
            prediction = None
        return ClassificationSample(
            id=sample_dict["id"],
            data=sample_dict["data"],
            external_id=sample_dict["externalId"] if "externalId" in sample_dict else None,
            annotation=annotation,
            prediction=prediction,
        )

    @staticmethod
    def label_from_dict(label_dict: Dict) -> ClassificationLabel:
        return ClassificationLabel(
            name=label_dict["name"],
            id=label_dict["id"],
            description=label_dict["description"] if "description" in label_dict else None,
            metadata=label_dict["metadata"] if "metadata" in label_dict else None,
        )

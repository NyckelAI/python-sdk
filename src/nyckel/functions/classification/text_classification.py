import time
from typing import Dict, List, Union

from nyckel.auth import OAuth2Renewer
from nyckel.functions.classification.classification import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationFunctionHandler,
    ClassificationFunctionURLHandler,
    ClassificationLabel,
    ClassificationLabelHandler,
    ClassificationPrediction,
    TextClassificationSample,
)
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, get_session_that_retries, repeated_get


class TextClassificationFunction(ClassificationFunction):
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth

        self._function_handler = ClassificationFunctionHandler(function_id, auth)
        self._label_handler = ClassificationLabelHandler(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()
        self._function_handler.validate_function()

    @classmethod
    def create_function(cls, name: str, auth: OAuth2Renewer) -> "TextClassificationFunction":
        return TextClassificationFunction(ClassificationFunctionHandler.create_function(name, "Text", auth), auth)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
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
            label_name=response.json()["labelName"],
            confidence=response.json()["confidence"],
        )

    @property
    def metrics(self) -> Dict:
        return self._function_handler.get_metrics()

    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        self._refresh_auth_token()
        if not self.has_trained_model():
            raise ValueError(
                f"No model trained yet for this function. Go to {self._url_handler.train_page} to see function status."
            )

        print(f"Invoking {len(sample_data_list)} text samples to {self._url_handler.train_page} ...")
        bodies = [{"data": sample_data} for sample_data in sample_data_list]
        endpoint = self._url_handler.api_endpoint("invoke")
        response_list = ParallelPoster(self._session, endpoint)(bodies)
        return [
            ClassificationPrediction(
                label_name=response.json()["labelName"],
                confidence=response.json()["confidence"],
            )
            for response in response_list
        ]

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        return self._label_handler.create_labels(labels)

    def list_labels(self) -> List[ClassificationLabel]:
        return self._label_handler.list_labels()

    def read_label(self, label_id: str) -> ClassificationLabel:
        return self._label_handler.read_label(label_id)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        return self._label_handler.update_label(label)

    def delete_label(self, label_id: str) -> None:
        return self._label_handler.delete_label(label_id)

    def create_samples(self, samples: List[TextClassificationSample]) -> List[str]:  # type: ignore
        self._refresh_auth_token()
        print(f"Posting {len(samples)} samples to {self._url_handler.train_page} ...")

        bodies = []
        for sample in samples:
            body: Dict[str, Union[str, Dict, None]] = {
                "data": sample.data,
                "externalId": sample.external_id,
            }
            if sample.annotation:
                body["annotation"] = {"labelName": sample.annotation.label_name}
            bodies.append(body)

        endpoint = self._url_handler.api_endpoint("samples")
        responses = ParallelPoster(self._session, endpoint)(bodies)
        return [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

    def list_samples(self) -> List[TextClassificationSample]:  # type: ignore
        self._refresh_auth_token()
        samples_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("samples"))
        labels = self._label_handler.list_labels()
        label_name_by_id = {label.id: label.name for label in labels}
        samples_typed = [self._sample_from_dict(entry, label_name_by_id) for entry in samples_dict_list]  # type: ignore
        return samples_typed

    def read_sample(self, sample_id: str) -> TextClassificationSample:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(f"samples/{sample_id}"))
        if response.status_code == 404:
            # If calling read right after create, the resource is not available yet. Sleep and retry once.
            time.sleep(1)
            response = self._session.get(self._url_handler.api_endpoint(f"samples/{sample_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"{response.status_code=}, {response.text=}. Unable to fetch sample {sample_id} from {self._url_handler.train_page}"
            )
        labels = self._label_handler.list_labels()
        label_name_by_id = {strip_nyckel_prefix(label.id): label.name for label in labels}  # type: ignore
        return self._sample_from_dict(response.json(), label_name_by_id)  # type: ignore

    def update_sample(self, sample: TextClassificationSample) -> TextClassificationSample:  # type: ignore
        raise NotImplementedError

    def delete_sample(self, sample_id: str) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint(f"samples/{sample_id}")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Sample {sample_id} deleted.")

    def delete(self) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint("")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Function {self._url_handler.train_page} deleted.")

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def _sample_from_dict(self, sample_dict: Dict, label_name_by_id: Dict[str, str]) -> TextClassificationSample:
        if "annotation" in sample_dict:
            annotation = ClassificationAnnotation(
                label_name=label_name_by_id[strip_nyckel_prefix(sample_dict["annotation"]["labelId"])],
            )
        else:
            annotation = None
        if "prediction" in sample_dict:
            prediction = ClassificationPrediction(
                confidence=sample_dict["prediction"]["confidence"],
                label_name=label_name_by_id[strip_nyckel_prefix(sample_dict["prediction"]["labelId"])],
            )
        else:
            prediction = None
        return TextClassificationSample(
            id=strip_nyckel_prefix(sample_dict["id"]),
            data=sample_dict["data"],
            external_id=sample_dict["externalId"] if "externalId" in sample_dict else None,
            annotation=annotation,
            prediction=prediction,
        )

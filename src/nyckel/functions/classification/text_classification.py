from typing import Dict, List

from nyckel.auth import OAuth2Renewer
from nyckel.auth import OAuth2Renewer
from nyckel.functions.classification.classification import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationFunctionURLHandler,
    ClassificationLabel,
    ClassificationPrediction,
    TextClassificationSample,
)

from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.classification.sample_handler import ClassificationSampleHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import get_session_that_retries, SequentialGetter


class TextClassificationFunction(ClassificationFunction):
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth

        self._function_handler = ClassificationFunctionHandler(function_id, auth)
        self._label_handler = ClassificationLabelHandler(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._sample_handler = ClassificationSampleHandler(function_id, auth)
        self._session = get_session_that_retries()
        self._refresh_auth_token()
        self._function_handler.validate_function("Text")

    @classmethod
    def create_function(cls, name: str, auth: OAuth2Renewer) -> "TextClassificationFunction":
        return TextClassificationFunction(ClassificationFunctionHandler.create_function(name, "Text", auth), auth)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        status_string = f"Name: {self.get_name()}, id: {self.function_id}, url: {self.train_page}"
        return status_string

    def __call__(self, sample_data: str) -> ClassificationPrediction:
        return self.invoke([sample_data])[0]

    @property
    def function_id(self) -> str:
        return self._function_id

    @property
    def sample_count(self) -> int:
        return self._function_handler.sample_count

    @property
    def label_count(self) -> int:
        return self._function_handler.label_count

    @property
    def train_page(self) -> str:
        return self._url_handler.train_page

    def get_name(self) -> str:
        return self._function_handler.get_name()

    @property
    def metrics(self) -> Dict:
        return self._function_handler.get_metrics()

    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        return self._sample_handler.invoke(sample_data_list, lambda x: x)

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

    def delete_labels(self, label_ids: List[str]) -> None:
        return self._label_handler.delete_labels(label_ids)

    def create_samples(self, samples: List[TextClassificationSample]) -> List[str]:  # type: ignore
        return self._sample_handler.create_samples(samples, lambda x: x)

    def list_samples(self) -> List[TextClassificationSample]:  # type: ignore
        self._refresh_auth_token()

        labels = self._label_handler.list_labels()
        label_name_by_id = {label.id: label.name for label in labels}
        samples_dict_list = SequentialGetter(self._session, self._url_handler.api_endpoint(path="samples"))()
        return [self._sample_from_dict(entry, label_name_by_id) for entry in samples_dict_list]  # type: ignore

    def read_sample(self, sample_id: str) -> TextClassificationSample:
        sample_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels()
        label_name_by_id = {strip_nyckel_prefix(label.id): label.name for label in labels}  # type: ignore

        return self._sample_from_dict(sample_dict, label_name_by_id)  # type: ignore

    def update_sample(self, sample: TextClassificationSample) -> TextClassificationSample:  # type: ignore
        raise NotImplementedError

    def delete_sample(self, sample_id: str) -> None:
        self._sample_handler.delete_sample(sample_id)

    def delete_samples(self, sample_ids: List[str]) -> None:
        self._sample_handler.delete_samples(sample_ids)

    def delete(self) -> None:
        self._function_handler.delete()

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

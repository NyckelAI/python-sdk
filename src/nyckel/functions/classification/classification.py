import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from nyckel import OAuth2Renewer


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
    data: Dict[str, Union[str, int]]  # Maps from field name (str) to field data (str or int)
    id: Optional[str] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


ClassificationSample = Union[TextClassificationSample, ImageClassificationSample, TabularClassificationSample]


class ClassificationFunction(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def new(cls, name: str, auth: OAuth2Renewer) -> "ClassificationFunction":
        pass

    @abc.abstractmethod
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        pass

    @abc.abstractmethod
    def __call__(self, sample_data: str) -> ClassificationPrediction:
        """Invokes the trained function. Raises ValueError if function is not trained"""
        pass

    @property
    @abc.abstractmethod
    def function_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def sample_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def label_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def train_page(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def input_modality(self) -> str:
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
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
    def delete_labels(self, label_ids: List[str]) -> None:
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
    def update_annotation(self, sample: ClassificationSample) -> None:
        pass

    @abc.abstractmethod
    def delete_sample(self, sample_id: str) -> None:
        pass

    @abc.abstractmethod
    def delete_samples(self, sample_ids: List[str]) -> None:
        pass

    @abc.abstractmethod
    def delete(self) -> None:
        """Deletes the function"""
        pass


class ClassificationFunctionURLHandler:
    def __init__(self, function_id: str, server_url: str):
        self._function_id = function_id
        self._server_url = server_url

    @property
    def train_page(self) -> str:
        return f"{self._server_url}/console/functions/{self._function_id}/train"

    def api_endpoint(self, path: Optional[str] = None, api_version: str = "v1", query_str: Optional[str] = None) -> str:
        endpoint = f"{self._server_url}/{api_version}/functions/{self._function_id}"
        if path:
            endpoint += f"/{path}"
        if query_str:
            endpoint += f"?{query_str}"
        return endpoint

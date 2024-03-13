import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from nyckel import Credentials, NyckelId

TextSampleData = str


ImageSampleData = str
"""DataUri, Url, or local filepath."""


TabularFieldValue = Union[str, float]
TabularFieldKey = str
TabularSampleData = Dict[TabularFieldKey, TabularFieldValue]

LabelName = str


@dataclass
class ClassificationLabel:
    name: LabelName
    id: Optional[NyckelId] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class ClassificationPrediction:
    label_name: LabelName
    confidence: float


@dataclass
class ClassificationAnnotation:
    label_name: str


@dataclass
class TabularFunctionField:
    name: str
    type: str
    """Type must be Number, Text or Image"""
    id: Optional[NyckelId] = None

    def __post_init__(self) -> None:
        assert self.type in ["Number", "Text", "Image"]


@dataclass
class ImageClassificationSample:
    data: ImageSampleData
    id: Optional[NyckelId] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


@dataclass
class TextClassificationSample:
    data: TextSampleData
    id: Optional[NyckelId] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


@dataclass
class TabularClassificationSample:
    data: TabularSampleData
    id: Optional[NyckelId] = None
    external_id: Optional[str] = None
    annotation: Optional[ClassificationAnnotation] = None
    prediction: Optional[ClassificationPrediction] = None


ClassificationSample = Union[TextClassificationSample, ImageClassificationSample, TabularClassificationSample]


class ClassificationFunction(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def create(cls, name: str, credentials: Credentials) -> "ClassificationFunction":
        pass

    @abc.abstractmethod
    def delete(self) -> None:
        pass

    @abc.abstractmethod
    def __init__(self, function_id: str, credentials: Credentials):
        pass

    @property
    @abc.abstractmethod
    def function_id(self) -> NyckelId:
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
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def invoke(self, sample_data_list: List[str], model_id: str) -> List[ClassificationPrediction]:
        """Invokes the trained function. Raises ValueError if function is not trained"""
        pass

    @abc.abstractmethod
    def has_trained_model(self) -> bool:
        pass

    @abc.abstractmethod
    def create_labels(self, labels: List[ClassificationLabel]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_labels(self) -> List[ClassificationLabel]:
        pass

    @abc.abstractmethod
    def read_label(self, label_id: NyckelId) -> ClassificationLabel:
        pass

    @abc.abstractmethod
    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        pass

    @abc.abstractmethod
    def delete_labels(self, label_ids: List[NyckelId]) -> None:
        pass

    @abc.abstractmethod
    def create_samples(self, samples: List[ClassificationSample]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_samples(self) -> List[ClassificationSample]:
        pass

    @abc.abstractmethod
    def read_sample(self, sample_id: NyckelId) -> ClassificationSample:
        pass

    @abc.abstractmethod
    def update_annotation(self, sample: ClassificationSample) -> None:
        pass

    @abc.abstractmethod
    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        pass


class ClassificationFunctionURLHandler:
    def __init__(self, function_id: NyckelId, server_url: str):
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

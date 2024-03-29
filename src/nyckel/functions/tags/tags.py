import abc
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from nyckel import Credentials, NyckelId
from nyckel.functions.classification.classification import (
    ClassificationLabel,
    ClassificationPrediction,
    ImageSampleData,
    TabularSampleData,
    TextSampleData,
)


@dataclass
class TagsAnnotation:
    label_name: str
    present: bool = True


TagsPrediction = Sequence[ClassificationPrediction]


@dataclass
class ImageTagsSample:
    data: ImageSampleData
    id: Optional[NyckelId] = None
    external_id: Optional[str] = None
    annotation: Optional[List[TagsAnnotation]] = None
    prediction: Optional[List[ClassificationPrediction]] = None


@dataclass
class TextTagsSample:
    data: TextSampleData
    id: Optional[NyckelId] = None
    external_id: Optional[str] = None
    annotation: Optional[List[TagsAnnotation]] = None
    prediction: Optional[List[ClassificationPrediction]] = None


@dataclass
class TabularTagsSample:
    data: TabularSampleData
    id: Optional[NyckelId] = None
    external_id: Optional[str] = None
    annotation: Optional[List[TagsAnnotation]] = None
    prediction: Optional[List[ClassificationPrediction]] = None


TagsSample = Union[TextTagsSample, ImageTagsSample, TabularTagsSample]


class TagsFunction(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def create(cls, name: str, credentials: Credentials) -> "TagsFunction":
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
    def invoke(self, sample_data_list: List[str]) -> List[TagsPrediction]:
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
    def create_samples(self, samples: List[TagsSample]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_samples(self) -> List[TagsSample]:
        pass

    @abc.abstractmethod
    def read_sample(self, sample_id: NyckelId) -> TagsSample:
        pass

    @abc.abstractmethod
    def update_annotation(self, sample: TagsSample) -> None:
        pass

    @abc.abstractmethod
    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        pass


class TagsFunctionURLHandler:
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

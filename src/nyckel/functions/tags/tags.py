from dataclasses import dataclass
from typing import List, Optional, Sequence

from nyckel.data_classes import NyckelId
from nyckel.functions.classification.classification import (
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

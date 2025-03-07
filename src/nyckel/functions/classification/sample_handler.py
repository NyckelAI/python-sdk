import time
from typing import Callable, Dict, List, Union

import requests
from tqdm import tqdm

from nyckel import (
    ClassificationPrediction,
    ClassificationPredictionError,
    ClassificationPredictionOrError,
    ClassificationSample,
    Credentials,
    ImageClassificationSample,
    TabularClassificationSample,
    TextClassificationSample,
)
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelDeleter, ParallelPoster, SequentialGetter
from nyckel.utils import chunkify_list

ClassificationSampleList = Union[
    List[TextClassificationSample], List[TabularClassificationSample], List[ImageClassificationSample]
]


class ClassificationSampleHandler:
    def __init__(self, function_id: str, credentials: Credentials):
        self._function_id = function_id
        self._credentials = credentials
        self._url_handler = ClassificationFunctionURLHandler(function_id, credentials.server_url)

    def invoke(
        self,
        sample_data_list: Union[List[Dict], List[str]],
        sample_data_transformer: Callable,
        model_id: str = "",
    ) -> List[ClassificationPredictionOrError]:

        bodies = [{"data": sample_data} for sample_data in sample_data_list]

        def body_transformer(body: Dict) -> Dict:
            body["data"] = sample_data_transformer(body["data"])
            return body

        session = self._credentials.get_session()
        endpoint = self._url_handler.api_endpoint(path="invoke", query_str=f"modelId={model_id}")
        progress_bar = tqdm(total=len(bodies), ncols=80, desc="Invoking")

        poster = ParallelPoster(session, endpoint, progress_bar, body_transformer)
        response_list = poster(bodies)
        parsed_responses = self.parse_predictions_response(response_list)

        return parsed_responses

    def parse_predictions_response(
        self, response_list: List[requests.Response]
    ) -> List[ClassificationPredictionOrError]:
        typed_responses: List[ClassificationPredictionOrError] = []
        for response in response_list:
            if response.status_code == 200:
                typed_responses.append(
                    ClassificationPrediction(
                        label_name=response.json()["labelName"],
                        confidence=response.json()["confidence"],
                    )
                )
            else:
                typed_responses.append(
                    ClassificationPredictionError(
                        error=response.text,
                        status_code=response.status_code,
                    )
                )
        return typed_responses

    def create_samples(self, samples: ClassificationSampleList, sample_data_transformer: Callable) -> List[str]:
        bodies = []
        for sample in samples:
            body = {"data": sample.data, "externalId": sample.external_id}
            if sample.annotation:
                body["annotation"] = {"labelName": sample.annotation.label_name}
            bodies.append(body)

        def body_transformer(body: Dict) -> Dict:
            body["data"] = sample_data_transformer(body["data"])
            return body

        session = self._credentials.get_session()
        url = self._url_handler.api_endpoint(path="samples")
        progress_bar = tqdm(total=len(bodies), ncols=80, desc="Posting samples")
        poster = ParallelPoster(session, url, progress_bar, body_transformer)
        response_list = []
        for chunk in chunkify_list(bodies, 500):
            response_list.extend(poster(chunk))
            poster.refresh_session(self._credentials.get_session())

        sample_ids = []
        for response in response_list:
            if response.status_code == 200:
                sample_ids.append(strip_nyckel_prefix(response.json()["id"]))
            if response.status_code == 409:
                sample_ids.append(strip_nyckel_prefix(response.json()["existingSampleId"]))

        return sample_ids

    def read_sample(self, sample_id: str) -> Dict:
        session = self._credentials.get_session()
        response = session.get(self._url_handler.api_endpoint(path=f"samples/{sample_id}"))
        if response.status_code == 404:
            # If calling read right after create, the resource is not available yet. Sleep and retry once.
            time.sleep(1)
            response = session.get(self._url_handler.api_endpoint(path=f"samples/{sample_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"{response.status_code=}, {response.text=}. Unable to fetch sample {sample_id} "
                f"from {self._url_handler.train_page}"
            )
        return response.json()

    def list_samples(self, sample_count: int) -> List[Dict]:
        session = self._credentials.get_session()
        samples_dict_list = SequentialGetter(
            session, self._url_handler.api_endpoint(path="samples?batchSize=1000&sortBy=creation&sortOrder=descending")
        )(tqdm(total=sample_count, ncols=80, desc="Listing samples"))

        return samples_dict_list

    def update_annotation(self, sample: ClassificationSample) -> None:
        url = self._url_handler.api_endpoint(path=f"samples/{sample.id}/annotation")
        session = self._credentials.get_session()
        if sample.annotation:
            body = {"labelName": sample.annotation.label_name}
            response = session.put(url, json=body)
            assert response.status_code == 200, f"Update failed with {response.status_code=}, {response.text=}"
        else:
            response = session.delete(url)
            assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"

    def delete_samples(self, sample_ids: List[str]) -> None:
        if len(sample_ids) == 0:
            return None
        session = self._credentials.get_session()
        sample_ids = [strip_nyckel_prefix(sample_id) for sample_id in sample_ids]
        parallel_deleter = ParallelDeleter(
            session, self._url_handler.api_endpoint(path="samples"), desc="Deleting samples"
        )
        parallel_deleter(sample_ids)

import time
from typing import Callable, Dict, List, Union


from nyckel.auth import OAuth2Renewer
from nyckel.functions.classification.classification import (
    ClassificationFunctionURLHandler,
    ClassificationPrediction,
    ImageClassificationSample,
    TabularClassificationSample,
    TextClassificationSample,
)
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelDeleter, ParallelPoster, get_session_that_retries
from nyckel.utils import chunkify_list

ClassificationSampleList = Union[
    List[TextClassificationSample], List[TabularClassificationSample], List[ImageClassificationSample]
]


class ClassificationSampleHandler:
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._session = get_session_that_retries()
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._refresh_auth_token()

    def invoke(
        self, sample_data_list: Union[List[Dict], List[str]], sample_data_transformer: Callable, chunk_size: int = 500
    ) -> List[ClassificationPrediction]:
        self._refresh_auth_token()
        predictions: List[ClassificationPrediction] = list()
        for chunk in chunkify_list(sample_data_list, 500):
            bodies = [{"data": sample_data_transformer(sample_data)} for sample_data in chunk]
            predictions.extend(self._invoke_chunk(bodies))
        return predictions

    def _invoke_chunk(self, invoke_bodies: List[Dict]) -> List[ClassificationPrediction]:
        endpoint = self._url_handler.api_endpoint(path="invoke")
        response_list = ParallelPoster(self._session, endpoint)(invoke_bodies)
        return [
            ClassificationPrediction(
                label_name=response.json()["labelName"],
                confidence=response.json()["confidence"],
            )
            for response in response_list
        ]

    def create_samples(
        self, samples: ClassificationSampleList, sample_data_transformer: Callable, chunk_size: int = 500
    ) -> List[str]:
        if len(samples) == 0:
            return []
        self._refresh_auth_token()
        sample_ids: List[str] = list()
        for samples_chunk in chunkify_list(samples, chunk_size):
            sample_ids.extend(self._create_samples_chunk(samples_chunk, sample_data_transformer))
        return sample_ids

    def _create_samples_chunk(self, samples: ClassificationSampleList, sample_data_transformer: Callable) -> List[str]:
        bodies = []
        for sample in samples:
            body = {"data": sample_data_transformer(sample.data), "externalId": sample.external_id}
            if sample.annotation:
                body["annotation"] = {"labelName": sample.annotation.label_name}
            bodies.append(body)

        endpoint = self._url_handler.api_endpoint(path="samples")
        responses = ParallelPoster(self._session, endpoint)(bodies)
        sample_ids = [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]
        return sample_ids

    def read_sample(self, sample_id: str) -> Dict:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(path=f"samples/{sample_id}"))
        if response.status_code == 404:
            # If calling read right after create, the resource is not available yet. Sleep and retry once.
            time.sleep(1)
            response = self._session.get(self._url_handler.api_endpoint(path=f"samples/{sample_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"{response.status_code=}, {response.text=}. Unable to fetch sample {sample_id} from {self._url_handler.train_page}"
            )
        return response.json()

    def delete_sample(self, sample_id: str) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint(path=f"samples/{strip_nyckel_prefix(sample_id)}")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"

    def delete_samples(self, sample_ids: List[str]) -> None:
        if len(sample_ids) == 0:
            return None
        self._refresh_auth_token()
        sample_ids = [strip_nyckel_prefix(sample_id) for sample_id in sample_ids]
        parallel_deleter = ParallelDeleter(self._session, self._url_handler.api_endpoint(path="samples"))
        parallel_deleter(sample_ids)

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

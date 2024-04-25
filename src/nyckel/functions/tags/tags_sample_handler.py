import time
from typing import Any, Callable, Dict, List, Tuple, Union

from tqdm import tqdm

from nyckel import (
    ClassificationPrediction,
    Credentials,
    ImageTagsSample,
    TabularTagsSample,
    TagsPrediction,
    TextTagsSample,
)
from nyckel.functions.tags.tags import TagsFunctionURLHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelDeleter, ParallelPoster, SequentialGetter
from nyckel.utils import chunkify_list

TagsSampleList = Union[List[TextTagsSample], List[ImageTagsSample], List[TabularTagsSample]]


class TagsSampleHandler:

    def __init__(self, function_id: str, credentials: Credentials) -> None:
        self._function_id = function_id
        self._credentials = credentials
        self._url_handler = TagsFunctionURLHandler(function_id, credentials.server_url)

    def invoke(
        self, sample_data_list: Union[List[str], List[Dict]], sample_data_transformer: Callable
    ) -> List[TagsPrediction]:
        n_max_attempt = 5
        for _ in range(n_max_attempt):
            invoke_ok, response_list = self._attempt_invoke(sample_data_list, sample_data_transformer)
            if invoke_ok:
                return self._parse_predictions_response(response_list)
            else:
                if "No model available to invoke function" in response_list[0].text:
                    print("Model not trained yet. Retrying...")
                else:
                    raise RuntimeError(f"Failed to invoke function. {response_list=}")
            time.sleep(5)
        raise TimeoutError("Still no model after {n_max_attempt} attempts. Please try again later.")

    def _attempt_invoke(
        self,
        sample_data_list: Union[List[Dict], List[str]],
        sample_data_transformer: Callable,
    ) -> Tuple[bool, List[Any]]:
        bodies = [{"data": sample_data} for sample_data in sample_data_list]

        def body_transformer(body: Dict) -> Dict:
            body["data"] = sample_data_transformer(body["data"])
            return body

        session = self._credentials.get_session()
        endpoint = self._url_handler.api_endpoint(path="invoke", api_version="v0.9")
        progress_bar = tqdm(total=len(bodies), ncols=80, desc="Invoking")

        poster = ParallelPoster(session, endpoint, progress_bar, body_transformer)
        response_list = poster(bodies)
        if response_list[0].status_code in [200]:
            return True, response_list
        else:
            return False, response_list

    def _parse_predictions_response(self, response_list: List[Any]) -> List[TagsPrediction]:
        tags_predictions: List[TagsPrediction] = []
        for response in response_list:
            tags_prediction = [
                ClassificationPrediction(
                    label_name=entry["labelName"],
                    confidence=entry["confidence"],
                )
                for entry in response.json()
            ]
            tags_predictions.append(tags_prediction)

        return tags_predictions

    def create_samples(self, samples: TagsSampleList, sample_data_transformer: Callable) -> List[str]:
        bodies = []
        for sample in samples:
            body = {"data": sample.data, "externalId": sample.external_id}
            if sample.annotation:
                body["annotation"] = [
                    {"labelName": ann.label_name, "present": ann.present} for ann in sample.annotation  # type: ignore
                ]
            bodies.append(body)

        def body_transformer(body: Dict) -> Dict:
            body["data"] = sample_data_transformer(body["data"])
            return body

        session = self._credentials.get_session()
        url = self._url_handler.api_endpoint(path="samples", api_version="v0.9")
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

    def list_samples(self, sample_count: int) -> List[Dict]:
        session = self._credentials.get_session()
        samples_dict_list = SequentialGetter(
            session,
            self._url_handler.api_endpoint(
                path="samples?batchSize=1000&sortBy=creation&sortOrder=descending", api_version="v0.9"
            ),
        )(tqdm(total=sample_count, ncols=80, desc="Listing samples"))

        return samples_dict_list

    def read_sample(self, sample_id: str) -> Dict:
        session = self._credentials.get_session()
        url = self._url_handler.api_endpoint(path=f"samples/{sample_id}", api_version="v0.9")
        response = session.get(url)
        if response.status_code == 404:
            # If calling read right after create, the resource is not available yet. Sleep and retry once.
            time.sleep(1)
            response = session.get(self._url_handler.api_endpoint(path=f"samples/{sample_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"{response.status_code=}, {response.text=}. Unable to fetch sample {sample_id} " f"from {url   }"
            )
        return response.json()

    def update_annotation(self, sample: Union[TextTagsSample, ImageTagsSample]) -> None:
        session = self._credentials.get_session()
        assert sample.annotation
        response = session.put(
            self._url_handler.api_endpoint(path=f"samples/{sample.id}/annotation", api_version="v0.9"),
            json=[
                {
                    "labelName": entry.label_name,
                    "present": entry.present,
                }
                for entry in sample.annotation
            ],
        )
        response.raise_for_status()

    def delete_samples(self, sample_ids: List[str]) -> None:
        if len(sample_ids) == 0:
            return None
        session = self._credentials.get_session()
        sample_ids = [strip_nyckel_prefix(sample_id) for sample_id in sample_ids]
        parallel_deleter = ParallelDeleter(
            session, self._url_handler.api_endpoint(path="samples", api_version="v0.9"), desc="Deleting samples"
        )
        parallel_deleter(sample_ids)

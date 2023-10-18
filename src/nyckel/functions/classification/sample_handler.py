import time
from typing import Any, Callable, Dict, List, Tuple, Union

from tqdm import tqdm

from nyckel import (
    ClassificationPrediction,
    ClassificationSample,
    ImageClassificationSample,
    TabularClassificationSample,
    TextClassificationSample,
    User,
)
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelDeleter, ParallelPoster, SequentialGetter, get_session_that_retries

ClassificationSampleList = Union[
    List[TextClassificationSample], List[TabularClassificationSample], List[ImageClassificationSample]
]


class ClassificationSampleHandler:
    def __init__(self, function_id: str, user: User):
        self._function_id = function_id
        self._user = user
        self._session = get_session_that_retries()
        self._url_handler = ClassificationFunctionURLHandler(function_id, user.server_url)
        self._refresh_auth_token()

    def invoke(
        self, sample_data_list: Union[List[Dict], List[str]], sample_data_transformer: Callable
    ) -> List[ClassificationPrediction]:
        n_max_attempt = 10
        for _ in range(n_max_attempt):
            invoke_ok, response_list = self.attempt_invoke(sample_data_list, sample_data_transformer)
            if invoke_ok:
                return self.parse_predictions_response(response_list)
            else:
                if "No model available to invoke function" in response_list[0].text:
                    print("Model not trained yet. Retrying...")
                else:
                    raise RuntimeError("Failed to invoke function. {response_list=}}")
            time.sleep(5)
        raise TimeoutError("Still no model after {n_max_attempt} attempts. Please try again later.")

    def attempt_invoke(
        self, sample_data_list: Union[List[Dict], List[str]], sample_data_transformer: Callable
    ) -> Tuple[bool, List[Any]]:
        self._refresh_auth_token()

        bodies = [{"data": sample_data} for sample_data in sample_data_list]
        endpoint = self._url_handler.api_endpoint(path="invoke")

        def body_transformer(body: Dict) -> Dict:
            body["data"] = sample_data_transformer(body["data"])
            return body

        poster = ParallelPoster(self._session, endpoint, desc="Invoking samples", body_transformer=body_transformer)
        response_list = poster(bodies)
        if response_list[0].status_code in [200]:
            return True, response_list
        else:
            return False, response_list

    def parse_predictions_response(self, response_list: List[Any]) -> List[ClassificationPrediction]:
        return [
            ClassificationPrediction(
                label_name=response.json()["labelName"],
                confidence=response.json()["confidence"],
            )
            for response in response_list
        ]

    def create_samples(self, samples: ClassificationSampleList, sample_data_transformer: Callable) -> List[str]:
        self._refresh_auth_token()

        bodies = []
        for sample in samples:
            body = {"data": sample.data, "externalId": sample.external_id}
            if sample.annotation:
                body["annotation"] = {"labelName": sample.annotation.label_name}
            bodies.append(body)

        endpoint = self._url_handler.api_endpoint(path="samples")

        def body_transformer(body: Dict) -> Dict:
            body["data"] = sample_data_transformer(body["data"])
            return body

        poster = ParallelPoster(self._session, endpoint, desc="Posting samples", body_transformer=body_transformer)
        response_list = poster(bodies)
        sample_ids = [
            strip_nyckel_prefix(response.json()["id"]) for response in response_list if response.status_code == 200
        ]
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
                f"{response.status_code=}, {response.text=}. Unable to fetch sample {sample_id} "
                f"from {self._url_handler.train_page}"
            )
        return response.json()

    def list_samples(self, sample_count: int) -> List[Dict]:
        self._refresh_auth_token()

        samples_dict_list = SequentialGetter(
            self._session, self._url_handler.api_endpoint(path="samples?batchSize=1000")
        )(tqdm(total=sample_count, ncols=80, desc="Listing samples"))

        return samples_dict_list

    def update_annotation(self, sample: ClassificationSample) -> None:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint(path=f"samples/{sample.id}/annotation")
        if sample.annotation:
            body = {"labelName": sample.annotation.label_name}
            response = self._session.put(url, json=body)
            assert response.status_code == 200, f"Update failed with {response.status_code=}, {response.text=}"
        else:
            response = self._session.delete(url)
            assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"

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
        parallel_deleter = ParallelDeleter(
            self._session, self._url_handler.api_endpoint(path="samples"), desc="Deleting samples"
        )
        parallel_deleter(sample_ids)

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._user.token})

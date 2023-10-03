import time
from typing import Dict, List, Optional

from tqdm import tqdm

from nyckel import OAuth2Renewer
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler, ClassificationLabel
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelDeleter, ParallelPoster, SequentialGetter, get_session_that_retries


class ClassificationLabelHandler:
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        self._refresh_auth_token()
        bodies = [
            {"name": label.name, "description": label.description, "metadata": label.metadata} for label in labels
        ]
        responses = ParallelPoster(self._session, self._url_handler.api_endpoint(path="labels"), desc="Posting labels")(
            bodies
        )

        label_ids = [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]
        self._confirm_new_labels_available(labels)
        return label_ids

    def _confirm_new_labels_available(self, new_labels: List[ClassificationLabel]) -> None:
        # Before returning, make sure the assets are available via the API.
        label_names_post_complete = False
        timeout_seconds = 5
        t0 = time.time()
        while not label_names_post_complete:
            if time.time() - t0 > timeout_seconds:
                raise ValueError("Something went wrong when posting labels.")
            time.sleep(0.5)
            labels_retrieved = self.list_labels(label_count=None)
            label_names_post_complete = set([label.name for label in new_labels]).issubset(
                [label.name for label in labels_retrieved]
            )

    def list_labels(self, label_count: Optional[int]) -> List[ClassificationLabel]:
        if label_count:
            progress_bar = tqdm(total=label_count, ncols=80, desc="Listing labels")
        else:
            progress_bar = None
        self._refresh_auth_token()
        labels_dict_list = SequentialGetter(self._session, self._url_handler.api_endpoint(path="labels"))(progress_bar)
        labels = [self._label_from_dict(entry) for entry in labels_dict_list]
        return labels

    def read_label(self, label_id: str) -> ClassificationLabel:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(path=f"labels/{label_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch label {label_id} from {self._url_handler.train_page} {response.text=} "
                f"{response.status_code=}"
            )
        label_dict = response.json()
        return self._label_from_dict(label_dict)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        assert label.id, "label to be updated must have the id field set"
        self._refresh_auth_token()
        response = self._session.put(
            self._url_handler.api_endpoint(path=f"labels/{strip_nyckel_prefix(label.id)}"),
            json={"name": label.name, "description": label.description, "metadata": label.metadata},
        )
        assert response.status_code == 200, f"Update failed with {response.status_code=}, {response.text=}"
        return label

    def delete_label(self, label_id: str) -> None:
        self._refresh_auth_token()
        response = self._session.delete(self._url_handler.api_endpoint(path=f"labels/{strip_nyckel_prefix(label_id)}"))
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"

    def delete_labels(self, label_ids: List[str]) -> None:
        if len(label_ids) == 0:
            return None
        self._refresh_auth_token()
        label_ids = [strip_nyckel_prefix(label_id) for label_id in label_ids]
        parallel_deleter = ParallelDeleter(self._session, self._url_handler.api_endpoint(path="labels"))
        parallel_deleter(label_ids)

    def _label_from_dict(self, label_dict: Dict) -> ClassificationLabel:
        return ClassificationLabel(
            name=label_dict["name"],
            id=strip_nyckel_prefix(label_dict["id"]),
            description=label_dict["description"] if "description" in label_dict else None,
            metadata=label_dict["metadata"] if "metadata" in label_dict else None,
        )

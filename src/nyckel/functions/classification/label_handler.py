import time
from typing import Dict, List, Optional

from tqdm import tqdm

from nyckel import ClassificationLabel, Credentials, NyckelId
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelDeleter, ParallelPoster, SequentialGetter


class ClassificationLabelHandler:
    def __init__(self, function_id: str, credentials: Credentials):
        self._function_id = function_id
        self._credentials = credentials
        self._url_handler = ClassificationFunctionURLHandler(function_id, credentials.server_url)

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        for label in labels:
            label.name = label.name.strip()
        session = self._credentials.get_session()
        bodies = [
            {"name": label.name, "description": label.description, "metadata": label.metadata} for label in labels
        ]
        url = self._url_handler.api_endpoint(path="labels")
        progress_bar = tqdm(total=len(bodies), ncols=80, desc="Posting labels")
        responses = ParallelPoster(session, url, progress_bar)(bodies)

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
        session = self._credentials.get_session()
        labels_dict_list = SequentialGetter(session, self._url_handler.api_endpoint(path="labels"))(progress_bar)
        labels = [self._label_from_dict(entry) for entry in labels_dict_list]
        return labels

    def read_label(self, label_id: NyckelId) -> ClassificationLabel:
        session = self._credentials.get_session()
        response = session.get(self._url_handler.api_endpoint(path=f"labels/{label_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch label {label_id} from {self._url_handler.train_page} {response.text=} "
                f"{response.status_code=}"
            )
        label_dict = response.json()
        return self._label_from_dict(label_dict)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        assert label.id, "label to be updated must have the id field set"
        session = self._credentials.get_session()
        response = session.put(
            self._url_handler.api_endpoint(path=f"labels/{strip_nyckel_prefix(label.id)}"),
            json={"name": label.name, "description": label.description, "metadata": label.metadata},
        )
        assert response.status_code == 200, f"Update failed with {response.status_code=}, {response.text=}"
        return label

    def delete_labels(self, label_ids: List[str]) -> None:
        if len(label_ids) == 0:
            return None
        label_ids = [strip_nyckel_prefix(label_id) for label_id in label_ids]
        session = self._credentials.get_session()
        parallel_deleter = ParallelDeleter(session, self._url_handler.api_endpoint(path="labels"))
        parallel_deleter(label_ids)

    def _label_from_dict(self, label_dict: Dict) -> ClassificationLabel:
        return ClassificationLabel(
            name=label_dict["name"],
            id=strip_nyckel_prefix(label_dict["id"]),
            description=label_dict["description"] if "description" in label_dict else None,
            metadata=label_dict["metadata"] if "metadata" in label_dict else None,
        )

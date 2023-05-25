import time
from typing import Dict, List

from nyckel import OAuth2Renewer
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler, ClassificationLabel
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, get_session_that_retries, repeated_get


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
        print(f"Posting {len(labels)} labels to {self._url_handler.train_page} ...")
        bodies = [
            {"name": label.name, "description": label.description, "metadata": label.metadata} for label in labels
        ]
        responses = ParallelPoster(self._session, self._url_handler.api_endpoint("labels"))(bodies)

        label_ids = [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

        # Before returning, make sure the assets are available via the API.
        label_names_post_complete = False
        while not label_names_post_complete:
            print("Waiting to confirm label assets are available...")
            time.sleep(0.5)
            labels_retrieved = self.list_labels()
            label_names_post_complete = set([l.name for l in labels]).issubset([l.name for l in labels_retrieved])

        return label_ids

    def list_labels(self) -> List[ClassificationLabel]:
        self._refresh_auth_token()
        labels_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("labels"))
        labels = [self._label_from_dict(entry) for entry in labels_dict_list]
        return labels

    def read_label(self, label_id: str) -> ClassificationLabel:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(f"labels/{label_id}"))
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch label {label_id} from {self._url_handler.train_page} {response.text=} {response.status_code=}"
            )
        label_dict = response.json()
        return self._label_from_dict(label_dict)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        raise NotImplementedError

    def delete_label(self, label_id: str) -> None:
        self._refresh_auth_token()
        response = self._session.delete(self._url_handler.api_endpoint(f"labels/{label_id}"))
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Label {label_id} deleted.")

    def _label_from_dict(self, label_dict: Dict) -> ClassificationLabel:
        return ClassificationLabel(
            name=label_dict["name"],
            id=strip_nyckel_prefix(label_dict["id"]),
            description=label_dict["description"] if "description" in label_dict else None,
            metadata=label_dict["metadata"] if "metadata" in label_dict else None,
        )

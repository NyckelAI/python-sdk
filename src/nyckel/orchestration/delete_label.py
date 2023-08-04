import sys
from typing import Dict, List

import fire  # type: ignore
from tqdm import tqdm

from nyckel import (
    ClassificationFunctionFactory,
    ClassificationLabel,
    OAuth2Renewer,
)
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.request_utils import SequentialGetter, get_session_that_retries


class NyckelLabelDeleter:
    def __init__(self, function_id: str, auth: OAuth2Renewer, skip_confirmation: bool = False):
        self._function_id = function_id
        self._auth = auth
        self._skip_confirmation = skip_confirmation
        self._function_handler = ClassificationFunctionHandler(self._function_id, self._auth)
        self._function = ClassificationFunctionFactory.load(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()

    def delete_label_and_samples(self, label_name: str) -> None:
        self._refresh_auth_token()
        function_labels = self._validate_label_name_exists(label_name)
        self._validate_function_type()
        samples_as_list_of_dicts = self._list_samples_for_label(label_name, function_labels)
        self._request_user_confirmation(len(samples_as_list_of_dicts), label_name)
        self._delete_samples([sample["id"] for sample in samples_as_list_of_dicts])
        self._delete_label(label_name, function_labels)

    def _validate_label_name_exists(self, label_name: str) -> List[ClassificationLabel]:
        labels = self._function.list_labels()
        if label_name not in [label.name for label in labels]:
            raise RuntimeError(f"Label: {label_name} not in function {self._function_id}")
        return labels

    def _validate_function_type(self) -> None:
        if not self._function_handler.get_output_modality() == "Classification":
            raise ValueError("Can only copy Classification functions.")
        if self._function_handler.get_input_modality() not in ["Text", "Image", "Tabular"]:
            raise ValueError(f"Unknown output type {self._function_handler.get_input_modality()}")

    def _request_user_confirmation(self, sample_count: int, label_name: str) -> None:
        if self._skip_confirmation:
            return None
        reply = input(
            f"-> This will delete label: '{label_name}' and the {sample_count} samples annotated as '{label_name}'. "
            f"Proceed (y/n)? "
        )
        if not reply == "y":
            print("-> Ok. Aborting...")
            sys.exit(0)

    def _list_samples_for_label(self, label_name: str, function_labels: List[ClassificationLabel]) -> List[Dict]:
        label_id_by_name = {label.name: label.id for label in function_labels}
        url = self._url_handler.api_endpoint(
            path="samples", query_str=f"annotationLabelId={label_id_by_name[label_name]}"
        )
        samples_list_dict = SequentialGetter(self._session, url)(tqdm(ncols=80, desc="Listing samples"))
        return samples_list_dict

    def _delete_samples(self, sample_ids: List[str]) -> None:
        self._function.delete_samples(sample_ids)
        print(f"-> {len(sample_ids)} samples deleted.")

    def _delete_label(self, label_name: str, function_labels: List[ClassificationLabel]) -> None:
        label_id_by_name = {label.name: label.id for label in function_labels}
        self._function.delete_label(label_id_by_name[label_name])  # type: ignore
        print(f"-> Label '{label_name}' deleted.")

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


def main(
    client_id: str, client_secret: str, function_id: str, label_name: str, server_url: str = "https://www.nyckel.com"
) -> None:
    auth = OAuth2Renewer(client_id, client_secret, server_url)
    label_deleter = NyckelLabelDeleter(function_id, auth)
    label_deleter.delete_label_and_samples(label_name)


if __name__ == "__main__":
    fire.Fire(main)

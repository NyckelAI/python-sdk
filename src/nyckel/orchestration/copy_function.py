import sys
from datetime import datetime
from typing import Type

import time
import fire

from nyckel import ImageClassificationFunction, OAuth2Renewer, TabularClassificationFunction, TextClassificationFunction
from nyckel.functions.classification.classification import ClassificationFunction
from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.request_utils import get_session_that_retries


class NyckelFunctionDuplicator:
    def __init__(self, from_function_id: str, auth: OAuth2Renewer, skip_confirmation: bool = False):
        self._from_function_id = from_function_id
        self._auth = auth
        self._skip_confirmation = skip_confirmation
        self._session = get_session_that_retries()
        self._function_handler: ClassificationFunctionHandler

    def __call__(self) -> ClassificationFunction:
        print(f"-> Starting copy process for {self._from_function_id} ...")
        self._refresh_auth_token()
        self._function_handler = ClassificationFunctionHandler(self._from_function_id, self._auth)
        self._validate_function_type()
        self._request_user_confirmation()
        return self._make_copy()

    def _validate_function_type(self) -> None:
        if not self._function_handler.get_output_modality() == "Classification":
            raise ValueError("Can only copy Classification functions.")
        if not self._function_handler.get_input_modality() in ["Text", "Image", "Tabular"]:
            raise ValueError(f"Unknown output type {self._function_handler.get_input_modality()}")

    def _get_function_type(self) -> Type[ClassificationFunction]:
        input_modality = self._function_handler.get_input_modality()
        if input_modality == "Text":
            return TextClassificationFunction
        if input_modality == "Image":
            return ImageClassificationFunction
        return TabularClassificationFunction

    def _request_user_confirmation(self) -> None:
        if self._skip_confirmation:
            return None
        reply = input(
            f"Ready to copy function: [name: {self._function_handler.get_name()}, id: {self._from_function_id}, label-count: {self._function_handler.label_count}, sample-count: {self._function_handler.sample_count}]. Ok to proceed (y/n)? "
        )
        if not reply == "y":
            print("Ok. Aborting...")
            sys.exit(0)

    def _make_copy(self) -> ClassificationFunction:
        print(f"-> Reading labels and samples from {self._from_function_id}")
        FunctionType = self._get_function_type()
        from_function = FunctionType(self._from_function_id, self._auth)
        labels = from_function.list_labels()
        samples = from_function.list_samples()

        to_function = FunctionType.create_function(self.new_function_name, self._auth)

        print(f"-> Adding labels and samples to {to_function.function_id}")
        to_function.create_labels(labels)
        to_function.create_samples(samples)
        time.sleep(2)  # Pause to allow assets to be available via the API.
        print(f"-> Done! New function available at: {to_function.train_page}")
        return to_function

    @property
    def new_function_name(self) -> str:
        return f"COPY OF {self._function_handler.get_name()} -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


def main(client_id: str, client_secret: str, from_function_id: str, server_url: str = "https://www.nyckel.com") -> None:
    auth = OAuth2Renewer(client_id, client_secret, server_url)
    duplicator = NyckelFunctionDuplicator(from_function_id, auth)
    duplicator()


if __name__ == "__main__":
    fire.Fire(main)

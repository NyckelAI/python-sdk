from datetime import datetime
from typing import Type

import fire

from nyckel import ImageClassificationFunction, OAuth2Renewer, TabularClassificationFunction, TextClassificationFunction
from nyckel.functions.classification.classification import ClassificationFunction
from nyckel.request_utils import get_session_that_retries


class NyckelFunctionDuplicator:
    def __init__(self, from_function_id: str, auth: OAuth2Renewer):
        self._from_function_id = from_function_id
        self._auth = auth
        self._session = get_session_that_retries()
        self._function_input_type: str

    def __call__(self) -> None:
        print(f"-> Starting copy process for {self._from_function_id} ...")
        self._refresh_auth_token()
        self._set_function_input_type()

        from_function = self._load_function()
        print(f"-> Loaded up function {self._from_function_id}")
        to_function = self._make_copy(from_function)
        if not to_function == from_function:
            print(f"-> Done!\n{from_function}\nsuccessfully copied to\n{to_function}")

    def _set_function_input_type(self) -> None:
        url = f"{self._auth.server_url}/v1/functions/{self._from_function_id}"
        response = self._session.get(url)
        assert response.status_code == 200, f"{response.text=} {response.status_code=}"
        if not response.json()["output"] == "Classification":
            raise ValueError("Can only copy classification functions.")
        function_input = response.json()["input"]
        assert function_input in ["Text", "Image", "Tabular"]
        self._function_input_type = function_input

    def _load_function(self) -> ClassificationFunction:
        return self._get_function_type()(self._from_function_id, self._auth)

    def _get_function_type(self) -> Type[ClassificationFunction]:
        if self._function_input_type == "Text":
            return TextClassificationFunction
        if self._function_input_type == "Image":
            return ImageClassificationFunction
        if self._function_input_type == "Tabular":
            return TabularClassificationFunction
        raise ValueError(f"Unknown input type {self._function_input_type=}")

    def _request_confirmation(self, function_name: str, function_id: str, n_labels: int, n_samples: int) -> bool:
        reply = input(
            f"Ok to proceed with copy of function: {function_name}, id: {function_id}, with {n_labels} labels and {n_samples} samples (y/n)? "
        )
        return reply == "y"

    def _make_copy(self, from_function: ClassificationFunction) -> ClassificationFunction:
        print(f"-> Reading labels and samples from {self._from_function_id}")
        labels = from_function.list_labels()
        samples = from_function.list_samples()

        if not self._request_confirmation(
            from_function.get_name(), from_function.function_id, len(labels), len(samples)
        ):
            print("Ok. Aborting...")
            return from_function

        new_function = self._get_function_type().create_function(
            f"{from_function.get_name()} [COPIED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", self._auth
        )

        print(f"-> Adding labels and samples to {new_function.function_id}")
        new_function.create_labels(labels)
        new_function.create_samples(samples)
        return new_function

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


def main(client_id: str, client_secret: str, from_function_id: str, server_url: str = "https://www.nyckel.com") -> None:
    auth = OAuth2Renewer(client_id, client_secret, server_url)
    duplicator = NyckelFunctionDuplicator(from_function_id, auth)
    duplicator()


if __name__ == "__main__":
    fire.Fire(main)

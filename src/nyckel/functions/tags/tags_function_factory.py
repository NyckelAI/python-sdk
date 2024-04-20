import time

from nyckel import Credentials
from nyckel.functions.tags import image_tags, tabular_tags, text_tags
from nyckel.functions.tags.tags_function_handler import TagsFunctionHandler
from nyckel.functions.utils import strip_nyckel_prefix


class TagsFunctionFactory:

    def __init__(self) -> None:
        self.function_type_by_input = {
            "Text": text_tags.TextTagsFunction,
            "Image": image_tags.ImageTagsFunction,
            "Tabular": tabular_tags.TabularTagsFunction,
        }

    @classmethod
    def load(self, function_id: str, credentials: Credentials):
        function_handler = TagsFunctionHandler(function_id, credentials)
        input_modality = function_handler.get_input_modality()
        return self.function_type_by_input[input_modality](function_id, credentials)

    def create(self, name: str, function_input: str, credentials: Credentials):
        def post_function() -> str:
            url = f"{credentials.server_url}/v1/functions"
            response = session.post(url, json={"input": function_input, "output": "Tags", "name": name})
            assert response.status_code == 200, f"Something went wrong when creating function: {response.text}"
            prefixed_function_id = response.json()["id"]
            function_id = strip_nyckel_prefix(prefixed_function_id)
            return function_id

        def hold_until_available(function_id: str) -> None:
            # Before returning, make sure the new function is available via the API.
            timeout_seconds = 5
            t0 = time.time()
            function_is_available = False
            while not function_is_available:
                if time.time() - t0 > timeout_seconds:
                    raise ValueError("Something went wrong when creating function.")
                time.sleep(0.25)
                url = f"{credentials.server_url}/v1/functions/{function_id}"
                response = session.get(url)
                function_is_available = response.status_code == 200

        session = credentials.get_session()
        function_id = post_function()
        hold_until_available(function_id)

        print(f"-> Created function {name} with id: {function_id}")
        return self.function_type_by_input[function_input](function_id, credentials)

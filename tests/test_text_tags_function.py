import unittest.mock
from typing import Dict

import requests
from nyckel import ClassificationPrediction, ClassificationPredictionError, Credentials, TagsPrediction
from nyckel.functions.tags.tags_sample_handler import TagsSampleHandler


def test_server_side_invoke_error_handling(auth_test_credentials: Credentials) -> None:

    # Mock the _post_as_json method to return empty response for "invalid_input"
    with unittest.mock.patch("nyckel.request_utils.ParallelPoster._post_as_json") as mock_post:

        def side_effect(data: Dict) -> requests.Response:
            response = requests.Response()
            if data.get("data") == "server side error":
                response.status_code = 400
                response._content = b'"Invalid input"'
                return response

            # Create successful response with JSON payload
            response.status_code = 200
            response._content = (
                b'[{"labelName": "Nice", "confidence": 0.9}, {"labelName": "Friendly", "confidence": 0.8}]'
            )
            return response

        mock_post.side_effect = side_effect
        sample_handler = TagsSampleHandler("function_id", auth_test_credentials)
        preds = sample_handler.invoke(["valid input", "server side error"], lambda x: x)
        assert len(preds) == 2
        assert len(preds[0]) == 2
        assert isinstance(preds[0][0], ClassificationPrediction)
        assert len(preds[1]) == 1
        assert isinstance(preds[1][0], ClassificationPredictionError)

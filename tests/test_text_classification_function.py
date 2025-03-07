import time
import unittest.mock
from typing import Dict, Tuple, Union

import pytest
import requests
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    ClassificationPredictionError,
    Credentials,
    TextClassificationFunction,
    TextClassificationSample,
)
from nyckel.functions.classification.sample_handler import ClassificationSampleHandler

post_sample_parameter_examples = [
    TextClassificationSample(data="Hi neighbor!", annotation=ClassificationAnnotation(label_name="Nice")),
    ("Hi neighbor!", "Nice"),
    "Hi neighbor!",
]


@pytest.mark.parametrize("post_samples_input", post_sample_parameter_examples)
def test_post_sample_overloading(
    text_classification_function: TextClassificationFunction,
    post_samples_input: Union[TextClassificationSample, Tuple[str, str], str],
) -> None:
    text_classification_function.create_samples([post_samples_input])
    time.sleep(1)
    samples = text_classification_function.list_samples()
    assert len(samples) == 1
    assert samples[0].data == "Hi neighbor!"


def test_samples(text_classification_function: TextClassificationFunction) -> None:
    func = text_classification_function
    label = ClassificationLabel(name="Nice")
    func.create_labels([label])

    sample = TextClassificationSample(data="hello", annotation=ClassificationAnnotation(label_name="Nice"))
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert sample_back.data == sample.data

    sample = TextClassificationSample(
        data="hello again", annotation=ClassificationAnnotation(label_name="Nice"), external_id="my_external_id"
    )
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert sample_back.data == sample.data
    assert sample_back.external_id == sample.external_id

    # Check that list_samples work
    samples_back = func.list_samples()
    assert len(samples_back) == 2
    assert set([sample.data for sample in samples_back]) == set(["hello", "hello again"])

    # Check delete
    func.delete_samples(sample_ids[0:1])
    time.sleep(1)
    samples_back = func.list_samples()
    assert len(samples_back) == 1
    assert samples_back[0].data == "hello"


def test_end_to_end(text_classification_function: TextClassificationFunction) -> None:
    func = text_classification_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)

    samples = [
        TextClassificationSample(data="hello", annotation=ClassificationAnnotation(label_name="Nice")),
        TextClassificationSample(data="hi", annotation=ClassificationAnnotation(label_name="Nice")),
        TextClassificationSample(data="Good bye", annotation=ClassificationAnnotation(label_name="Boo")),
        TextClassificationSample(data="I'm leaving", annotation=ClassificationAnnotation(label_name="Boo")),
        TextClassificationSample(data="Hi again"),
    ]

    func.create_samples(samples)

    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    assert isinstance(func.invoke(["Howdy"])[0], ClassificationPrediction)

    assert len(func.invoke(["Hej", "Hola", "Gruezi"])) == 3

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], TextClassificationSample)


def test_server_side_invoke_error_handling(auth_test_credentials: Credentials) -> None:

    # Mock the _post_as_json method to return empty response for "invalid_input"
    with unittest.mock.patch("nyckel.request_utils.ParallelPoster._post_as_json") as mock_post:

        def side_effect(data: Dict) -> requests.Response:
            response = requests.Response()
            if data.get("data") == "server side error":
                response.status_code = 400
                response._content = b'{"error": "Invalid input"}'
                return response

            # Create successful response with JSON payload
            response.status_code = 200
            response._content = b'{"labelName": "Nice", "confidence": 0.9}'
            return response

        mock_post.side_effect = side_effect
        preds = ClassificationSampleHandler("fid", auth_test_credentials).invoke(
            ["valid input", "server side error"], lambda x: x
        )

        assert len(preds) == 2
        assert isinstance(preds[0], ClassificationPrediction)
        assert isinstance(preds[1], ClassificationPredictionError)
        assert preds[1].status_code == 400


def test_client_side_invoke_error_handling(auth_test_credentials: Credentials) -> None:

    with unittest.mock.patch("requests.Session.post") as mock_post:

        def side_effect(url: str, json: Dict) -> requests.Response:
            response = requests.Response()
            if json.get("data") == "SDK side error":
                raise Exception("Client side error")
            else:
                response.status_code = 200
                response._content = b'{"labelName": "Nice", "confidence": 0.9}'
            return response

        mock_post.side_effect = side_effect

        preds = ClassificationSampleHandler("fid", auth_test_credentials).invoke(
            ["valid input", "SDK side error"], lambda x: x
        )

        assert len(preds) == 2
        assert isinstance(preds[0], ClassificationPrediction)
        assert isinstance(preds[1], ClassificationPredictionError)
        assert preds[1].status_code == 400

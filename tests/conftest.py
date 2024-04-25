import os
import random
import string
import time
from typing import Iterator

import numpy as np
import pytest
import requests
from nyckel import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationLabel,
    Credentials,
    ImageClassificationFunction,
    ImageEncoder,
    ImageTagsFunction,
    TabularClassificationFunction,
    TabularClassificationSample,
    TabularTagsFunction,
    TabularTagsSample,
    TagsAnnotation,
    TextClassificationFunction,
    TextClassificationSample,
    TextTagsFunction,
)
from nyckel.functions.classification.classification import TabularFunctionField
from PIL import Image

standard_tabular_test_fields = [
    TabularFunctionField(name="name", type="Text"),
    TabularFunctionField(name="age", type="Number"),
    TabularFunctionField(name="mug", type="Image"),
]

small_image_url = "https://www.nyckel.com/favicon.png"
large_image_url = "https://www.nyckel.com/favicon-500.png"


def make_random_image(size: int = 100) -> str:
    imarray = np.random.rand(size, size, 3) * 255
    img = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    return ImageEncoder().to_base64(img)


def make_random_text() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=50))


def make_random_tabular() -> dict:
    return {
        "name": make_random_text(),
        "age": random.randint(0, 100),
        "mug": make_random_image(),
    }


def hold_until_list_samples_available(function: ClassificationFunction, expected_count: int) -> None:
    actual_count = 0
    while not actual_count == expected_count:
        actual_count = len(function.list_samples())
        time.sleep(0.25)


def hold_until_function_trained(func: ClassificationFunction) -> None:
    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)


@pytest.fixture
def auth_test_credentials() -> Iterator[Credentials]:
    credentials = get_test_credentials()
    assert_credentials_has_access(credentials)
    if not credentials_has_project(credentials):
        create_project_for_credentials(credentials)
    yield credentials


@pytest.fixture
def text_classification_function(auth_test_credentials: Credentials) -> Iterator[TextClassificationFunction]:
    func = TextClassificationFunction.create("PYTHON-SDK TEXT TEST FUNCTION", auth_test_credentials)
    yield func
    func.delete()


@pytest.fixture
def text_classification_function_with_content(
    auth_test_credentials: Credentials,
) -> Iterator[TextClassificationFunction]:
    func = TextClassificationFunction.create("PYTHON-SDK TEXT TEST FUNCTION", auth_test_credentials)
    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)
    nice = ClassificationAnnotation(label_name="Nice")
    boo = ClassificationAnnotation(label_name="Boo")
    samples = [
        TextClassificationSample(data="hello", external_id="1", annotation=nice),
        TextClassificationSample(data="hi", external_id="2", annotation=nice),
        TextClassificationSample(data="Good bye", external_id="3", annotation=boo),
        TextClassificationSample(data="I'm leaving", external_id="4", annotation=boo),
        TextClassificationSample(data="Hi again", external_id="5"),
    ]
    func.create_samples(samples)
    hold_until_list_samples_available(func, len(samples))
    yield func
    func.delete()


@pytest.fixture
def image_classification_function(auth_test_credentials: Credentials) -> Iterator[ImageClassificationFunction]:
    func = ImageClassificationFunction.create("PYTHON-SDK IMAGE TEST FUNCTION", auth_test_credentials)
    yield func
    func.delete()


@pytest.fixture
def tabular_classification_function(auth_test_credentials: Credentials) -> Iterator[TabularClassificationFunction]:
    func = TabularClassificationFunction.create("PYTHON-SDK TABULAR TEST FUNCTION", auth_test_credentials)
    yield func
    func.delete()


@pytest.fixture
def tabular_classification_function_with_fields(auth_test_credentials: Credentials) -> Iterator[ImageTagsFunction]:
    func = TabularClassificationFunction.create("PYTHON-SDK TABULAR TEST FUNCTION", auth_test_credentials)
    func.create_fields(standard_tabular_test_fields)
    yield func
    func.delete()


@pytest.fixture
def trained_tabular_classification_function(
    auth_test_credentials: Credentials,
) -> Iterator[TabularClassificationFunction]:
    func = TabularClassificationFunction.create("PYTHON-SDK TABULAR TEST FUNCTION", auth_test_credentials)

    func.create_fields(standard_tabular_test_fields)
    samples = [
        TabularClassificationSample(data=make_random_tabular(), annotation=ClassificationAnnotation(label_name="a")),
        TabularClassificationSample(data=make_random_tabular(), annotation=ClassificationAnnotation(label_name="a")),
        TabularClassificationSample(data=make_random_tabular(), annotation=ClassificationAnnotation(label_name="b")),
        TabularClassificationSample(data=make_random_tabular(), annotation=ClassificationAnnotation(label_name="b")),
    ]
    func.create_samples(samples)
    hold_until_function_trained(func)
    yield func
    func.delete()


@pytest.fixture
def text_tags_function(auth_test_credentials: Credentials) -> Iterator[TextTagsFunction]:
    func = TextTagsFunction.create("PYTHON-SDK TEXT TAGS TEST FUNCTION", auth_test_credentials)
    yield func
    func.delete()


@pytest.fixture
def image_tags_function(auth_test_credentials: Credentials) -> Iterator[ImageTagsFunction]:
    func = ImageTagsFunction.create("PYTHON-SDK IMAGE TAGS TEST FUNCTION", auth_test_credentials)
    yield func
    func.delete()


@pytest.fixture
def tabular_tags_function(auth_test_credentials: Credentials) -> Iterator[TabularTagsFunction]:
    func = TabularTagsFunction.create("PYTHON-SDK TABULAR TAGS TEST FUNCTION", auth_test_credentials)
    yield func
    func.delete()


@pytest.fixture
def tabular_tags_function_with_fields(auth_test_credentials: Credentials) -> Iterator[TabularTagsFunction]:
    func = TabularTagsFunction.create("PYTHON-SDK TABULAR TAGS TEST FUNCTION", auth_test_credentials)
    func.create_fields(standard_tabular_test_fields)
    yield func
    func.delete()


@pytest.fixture
def trained_tabular_tags_function(auth_test_credentials: Credentials) -> Iterator[TabularTagsFunction]:
    func = TabularTagsFunction.create("PYTHON-SDK TABULAR TAGS TEST FUNCTION", auth_test_credentials)
    func.create_fields(standard_tabular_test_fields)
    samples = [
        TabularTagsSample(data=make_random_tabular(), annotation=[TagsAnnotation(label_name="a")]),
        TabularTagsSample(data=make_random_tabular(), annotation=[TagsAnnotation(label_name="a")]),
        TabularTagsSample(data=make_random_tabular(), annotation=[TagsAnnotation(label_name="b")]),
        TabularTagsSample(data=make_random_tabular(), annotation=[TagsAnnotation(label_name="b")]),
    ]
    func.create_samples(samples)
    hold_until_function_trained(func)
    yield func
    func.delete()


def get_test_credentials() -> Credentials:
    if "NYCKEL_PYTHON_SDK_CLIENT_SECRET" in os.environ:
        credentials = Credentials(
            client_id="python-sdk-test",
            client_secret=os.environ["NYCKEL_PYTHON_SDK_CLIENT_SECRET"],
            server_url="https://www.nyckel-staging.com",
        )
    else:
        credentials = Credentials(
            client_id="python-sdk-test",
            client_secret="secret",
            server_url="http://localhost:5000",
        )
    return credentials


def assert_credentials_has_access(credentials: Credentials) -> None:
    credentials.token


def credentials_has_project(credentials: Credentials) -> bool:
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + credentials.token})
    response = session.get(f"{credentials.server_url}/v0.9/projects")
    assert response.status_code == 200
    return len(response.json()) > 0


def create_project_for_credentials(credentials: Credentials) -> None:
    session = credentials.get_session()
    print("-> Creating project for credentials")
    response = session.post(f"{credentials.server_url}/v0.9/projects", json={"name": "my_project"})
    assert response.status_code == 200

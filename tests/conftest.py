import os
import time
from typing import Iterator

import numpy as np
import pytest
import requests
from nyckel import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationLabel,
    ImageClassificationFunction,
    ImageClassificationSample,
    TabularClassificationFunction,
    TabularClassificationSample,
    TextClassificationFunction,
    TextClassificationSample,
    User,
)
from nyckel.functions.classification.image_classification import ImageEncoder
from PIL import Image


def make_random_image(size: int = 100) -> str:
    imarray = np.random.rand(size, size, 3) * 255
    img = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    return ImageEncoder().image_to_base64(img)


def hold_until_list_samples_available(function: ClassificationFunction, expected_count: int) -> None:
    actual_count = 0
    while not actual_count == expected_count:
        actual_count = len(function.list_samples())
        time.sleep(0.25)


@pytest.fixture
def auth_test_user() -> Iterator[User]:
    user = get_test_user()
    assert_user_has_access(user)
    if not user_has_project(user):
        create_project_for_user(user)
    yield user


@pytest.fixture
def text_classification_function(auth_test_user: User) -> Iterator[TextClassificationFunction]:
    func = TextClassificationFunction.create_function("PYTHON-SDK TEXT TEST FUNCTION", auth_test_user)
    yield func
    func.delete()


@pytest.fixture
def text_classification_function_with_content(auth_test_user: User) -> Iterator[TextClassificationFunction]:
    func = TextClassificationFunction.create_function("PYTHON-SDK TEXT TEST FUNCTION", auth_test_user)
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
def image_classification_function(auth_test_user: User) -> Iterator[ImageClassificationFunction]:
    func = ImageClassificationFunction.create_function("PYTHON-SDK IMAGE TEST FUNCTION", auth_test_user)
    yield func
    func.delete()


@pytest.fixture
def image_classification_function_with_content(auth_test_user: User) -> Iterator[ImageClassificationFunction]:
    func = ImageClassificationFunction.create_function("PYTHON-SDK IMAGE TEST FUNCTION", auth_test_user)
    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)
    nice = ClassificationAnnotation(label_name="Nice")
    boo = ClassificationAnnotation(label_name="Boo")
    samples = [
        ImageClassificationSample(data=make_random_image(), external_id="1", annotation=nice),
        ImageClassificationSample(data=make_random_image(), external_id="2", annotation=nice),
        ImageClassificationSample(data=make_random_image(), external_id="3", annotation=boo),
        ImageClassificationSample(data=make_random_image(), external_id="4", annotation=boo),
        ImageClassificationSample(data=make_random_image(), external_id="5"),
    ]
    func.create_samples(samples)
    hold_until_list_samples_available(func, len(samples))
    yield func
    func.delete()


@pytest.fixture
def tabular_classification_function(auth_test_user: User) -> Iterator[TabularClassificationFunction]:
    func = TabularClassificationFunction.create_function("PYTHON-SDK TABULAR TEST FUNCTION", auth_test_user)
    yield func
    func.delete()


@pytest.fixture
def tabular_classification_function_with_content(
    auth_test_user: User,
) -> Iterator[TabularClassificationFunction]:
    func = TabularClassificationFunction.create_function("PYTHON-SDK TABULAR TEST FUNCTION", auth_test_user)
    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)
    nice = ClassificationAnnotation(label_name="Nice")
    boo = ClassificationAnnotation(label_name="Boo")
    samples = [
        TabularClassificationSample(data={"firstname": "Adam", "lastname": "Adams"}, external_id="1", annotation=nice),
        TabularClassificationSample(data={"firstname": "Bo", "lastname": "Biden"}, external_id="2", annotation=nice),
        TabularClassificationSample(data={"firstname": "Carl", "lastname": "Carrot"}, external_id="3", annotation=boo),
        TabularClassificationSample(data={"firstname": "Dick", "lastname": "Denali"}, external_id="4", annotation=boo),
        TabularClassificationSample(data={"firstname": "Erik", "lastname": "Elk"}, external_id="5"),
    ]
    func.create_samples(samples)
    hold_until_list_samples_available(func, len(samples))
    yield func
    func.delete()


def get_test_user() -> User:
    if "NYCKEL_PYTHON_SDK_CLIENT_SECRET" in os.environ:
        user = User(
            client_id="python-sdk-test",
            client_secret=os.environ["NYCKEL_PYTHON_SDK_CLIENT_SECRET"],
            server_url="https://www.nyckel-staging.com",
        )
    else:
        user = User(
            client_id="python-sdk-test",
            client_secret="secret",
            server_url="http://localhost:5000",
        )
    return user


def assert_user_has_access(user: User) -> None:
    user.token


def user_has_project(user: User) -> bool:
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + user.token})
    response = session.get(f"{user.server_url}/v0.9/projects")
    assert response.status_code == 200
    return len(response.json()) > 0


def create_project_for_user(user: User) -> None:
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + user.token})
    print("-> Creating project for user")
    response = session.post(f"{user.server_url}/v0.9/projects", json={"name": "my_project"})
    assert response.status_code == 200

import os

import pytest
import requests
from nyckel import ImageClassificationFunction, OAuth2Renewer, TextClassificationFunction, TabularClassificationFunction


@pytest.fixture
def auth_test_user() -> OAuth2Renewer:
    user = get_test_user()
    assert_user_has_access(user)
    if not user_has_project(user):
        create_project_for_user(user)
    yield user


@pytest.fixture
def text_classification_function(auth_test_user):
    func = TextClassificationFunction.create_function("PYTHON-SDK TEXT TEST FUNCTION", auth_test_user)
    yield func
    func.delete()


@pytest.fixture
def image_classification_function(auth_test_user):
    func = ImageClassificationFunction.create_function("PYTHON-SDK IMAGE TEST FUNCTION", auth_test_user)
    yield func
    func.delete()


@pytest.fixture
def tabular_classification_function(auth_test_user):
    func = TabularClassificationFunction.create_function("PYTHON-SDK TABULAR TEST FUNCTION", auth_test_user)
    yield func
    # func.delete()


def get_test_user() -> OAuth2Renewer:
    if "NYCKEL_PYTHON_SDK_CLIENT_SECRET" in os.environ:
        auth = OAuth2Renewer(
            client_id="python-sdk-test",
            client_secret=os.environ["NYCKEL_PYTHON_SDK_CLIENT_SECRET"],
            server_url="https://www.nyckel-staging.com",
        )
    else:
        auth = OAuth2Renewer(
            client_id="python-sdk-test",
            client_secret="secret",
            server_url="http://localhost:5000",
        )
    return auth


def assert_user_has_access(user: OAuth2Renewer):
    try:
        user.token
    except:
        raise RuntimeError(f"Test client can not connect to {user.server_url}")


def user_has_project(user: OAuth2Renewer):
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + user.token})
    response = session.get(f"{user.server_url}/v0.9/projects")
    assert response.status_code == 200
    return len(response.json()) > 0


def create_project_for_user(user: OAuth2Renewer):
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + user.token})
    print("Creating project for user")
    response = session.post(f"{user.server_url}/v0.9/projects", json={"name": "my_project"})
    assert response.status_code == 200

import os

import pytest
import requests
from nyckel import OAuth2Renewer


@pytest.fixture(scope="session", autouse=True)
def auth_test_user():
    user = get_test_user()
    assert_user_has_access(user)
    create_project_for_user(user)
    yield user
    # clean_up_functions(user)


def clean_up_functions(user: OAuth2Renewer):
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + user.token})
    response = session.get(f"{user.server_url}/v0.9/projects")
    assert response.status_code == 200
    for function_id in response.json()[0]["functionIds"]:
        response = session.delete(f"{user.server_url}/v0.9/functions/{function_id}")
        print(f"{response.status_code=}")
        print(f"Cleaned up {function_id}")


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


def create_project_for_user(user: OAuth2Renewer):
    session = requests.Session()
    session.headers.update({"authorization": "Bearer " + user.token})
    response = session.get(f"{user.server_url}/v0.9/projects")
    assert response.status_code == 200
    if len(response.json()) == 0:
        print("Creating project for user")
        response = session.post(f"{user.server_url}/v0.9/projects", json={"name": "my_project"})
        assert response.status_code == 200
    else:
        pass

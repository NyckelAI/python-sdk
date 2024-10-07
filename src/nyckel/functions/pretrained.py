from typing import Dict

from nyckel import Credentials


def invoke(function_id: str, data: str, credentials: Credentials) -> Dict:
    session = credentials.get_session()
    endpoint = f"https://www.nyckel.com/v1/functions/{function_id}/invoke"
    return session.post(endpoint, json={"data": data}).json()

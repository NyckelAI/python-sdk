from typing import Dict

from nyckel import Credentials


def invoke(model_id: str, data: str, client_id: str, client_secret: str) -> Dict:
    session = Credentials(client_id, client_secret).get_session()
    endpoint = f"https://www.nyckel.com/v1/functions/{model_id}/invoke"
    return session.post(endpoint, json={"data": data}).json()

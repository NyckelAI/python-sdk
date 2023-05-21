import concurrent.futures
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from nyckel.config import NBR_CONCURRENT_REQUESTS


class ParallelPoster:
    def __init__(self, session: requests.Session, endpoint: str):
        self._session = session
        self._endpoint = endpoint

    def _post_as_json(self, data: Dict) -> requests.Response:
        response = self._session.post(self._endpoint, json=data)
        return response

    def __call__(self, bodies: List[Dict]) -> List[requests.Response]:
        if len(bodies) == 0:
            return []
        responses = [requests.Response()] * len(bodies)
        n_workers = min(len(bodies), NBR_CONCURRENT_REQUESTS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            index_by_future = {executor.submit(self._post_as_json, body): index for index, body in enumerate(bodies)}
            for future in tqdm(concurrent.futures.as_completed(index_by_future)):
                index = index_by_future[future]
                body = bodies[index]
                response = future.result()
                if response.status_code == 200:
                    pass  # It worked.
                else:
                    raise ValueError(
                        f"Posting {body} to {self._endpoint} failed with {response.status_code=} {response.text=}"
                    )
                responses[index] = response
        return responses


def repeated_get(session: requests.Session, endpoint: str) -> List[Dict]:
    def get_base_url(endpoint: str) -> Tuple[str, str]:
        if ":5000" in endpoint:
            # Running locally
            parts = endpoint.split(":5000")
            base_url, slug = parts
            base_url += ":5000"
        else:
            parts = endpoint.split(".com")
            base_url, slug = parts
            base_url += ".com"
        return base_url, slug

    base_url, slug = get_base_url(endpoint)
    resp = session.get(base_url + slug)
    if not resp.status_code == 200:
        raise RuntimeError(f"GET from {base_url+slug} failed with {resp.status_code}, {resp.text}.")
    resource_list = resp.json()
    while "next" in resp.links:
        slug = resp.links["next"]["url"]
        resp = session.get(base_url + slug)
        if not resp.status_code == 200:
            raise RuntimeError(f"GET from {base_url+slug} failed with {resp.status_code}, {resp.text}.")
        resource_list.extend(resp.json())

    return resource_list


def get_session_that_retries() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session

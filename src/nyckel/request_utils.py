import concurrent.futures
from typing import Dict, List

import requests
from tqdm import tqdm

from nyckel.config import NBR_CONCURRENT_REQUESTS


class ParallelPoster:
    def __init__(self, session: requests.Session, endpoint: str):
        self._session = session
        self._endpoint = endpoint

    def _post_as_json(self, data: Dict):
        response = self._session.post(self._endpoint, json=data)
        return response

    def __call__(self, bodies: List[Dict]) -> List[requests.Response]:
        responses = [requests.Response()] * len(bodies)
        n_workers = min(len(bodies), NBR_CONCURRENT_REQUESTS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            index_by_future = {executor.submit(self._post_as_json, body): index for index, body in enumerate(bodies)}
            for future in tqdm(concurrent.futures.as_completed(index_by_future)):
                index = index_by_future[future]
                body = bodies[index]
                response = future.result()
                if response.status_code not in [200, 409]:
                    print(f"Posting {body} to {self._endpoint} failed with {response.status_code=} {response.text=}")
                else:
                    pass  # It worked.
                responses[index] = response
        return responses


def repeated_get(session: requests.Session, endpoint: str):
    base_url, slug = endpoint.split(".com")
    base_url += ".com"
    resp = session.get(base_url + slug)
    resource_list = resp.json()
    while "next" in resp.links:
        slug = resp.links["next"]["url"]
        response = session.get(base_url + slug)
        assert resp.status_code == 200, f"GET from {base_url+slug} failed with {response.status_code}, {response.text}."
        resource_list.extend(resp.json())

    return resource_list

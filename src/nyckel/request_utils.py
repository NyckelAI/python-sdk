import concurrent.futures
import warnings
from json import JSONDecodeError
from typing import Callable, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from nyckel.config import NBR_CONCURRENT_REQUESTS


class ParallelPoster:
    def __init__(
        self,
        session: requests.Session,
        endpoint: str,
        progress_bar: Optional[tqdm] = None,
        body_transformer: Callable = lambda x: x,
    ):
        """last input 'body_transformer' intended for image functions,
        where we want to resize and such at the very last instance before posting.
        This is to avoid blowing up the memory for large sets of images to post."""
        self._session = session
        self._endpoint = endpoint
        self._body_transformer = body_transformer
        if progress_bar is not None:
            self.progress_bar = progress_bar

        else:
            self.progress_bar = tqdm("Posting", ncols=80)

    def _post_as_json(self, data: Dict) -> requests.Response:
        response = self._session.post(self._endpoint, json=self._body_transformer(data))
        return response

    def refresh_session(self, session: requests.Session) -> None:
        self._session = session

    def __call__(self, bodies: List[Dict]) -> List[requests.Response]:
        if len(bodies) == 0:
            return []
        responses = [requests.Response()] * len(bodies)
        n_workers = min(len(bodies), NBR_CONCURRENT_REQUESTS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            index_by_future = {executor.submit(self._post_as_json, body): index for index, body in enumerate(bodies)}
            for future in concurrent.futures.as_completed(index_by_future):
                index = index_by_future[future]
                body = bodies[index]
                self.progress_bar.update(1)
                try:
                    response = future.result()
                    if response.status_code not in [200, 409]:
                        warnings.warn(
                            f"Posting {body} to {self._endpoint} failed with {response.status_code=} {response.text=}",
                            RuntimeWarning,
                        )
                    responses[index] = response
                except Exception as e:
                    warnings.warn(f"Posting {body} to {self._endpoint} failed with {e}", RuntimeWarning)

        return responses


class ParallelDeleter:
    def __init__(
        self,
        session: requests.Session,
        endpoint: str,
        desc: Optional[str] = None,
    ):
        self._session = session
        self._endpoint = endpoint
        self._desc = desc

    def _delete_one(self, asset_id: str) -> requests.Response:
        response = self._session.delete(f"{self._endpoint}/{asset_id}")
        return response

    def __call__(self, asset_ids: List[str]) -> List[requests.Response]:
        responses = [requests.Response()] * len(asset_ids)
        n_workers = min(len(asset_ids), NBR_CONCURRENT_REQUESTS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            index_by_future = {
                executor.submit(self._delete_one, asset_id): index for index, asset_id in enumerate(asset_ids)
            }
            for future in tqdm(
                concurrent.futures.as_completed(index_by_future), total=len(asset_ids), desc=self._desc, ncols=80
            ):
                index = index_by_future[future]
                asset_id = asset_ids[index]
                response = future.result()
                if not response.status_code == 200:
                    raise ValueError(
                        f"Error when deleting asset: {self._endpoint}/{asset_id}. {response.status_code=} "
                        f"{response.text=}"
                    )
                responses[index] = response
        return responses


class SequentialGetter:
    def __init__(self, session: requests.Session, endpoint: str):
        self._session = session
        self._endpoint = endpoint

    def __call__(self, progress_bar: Optional[tqdm] = None) -> List[Dict]:
        base_url, slug = self._get_base_url()

        resp = self._session.get(base_url + slug)
        if not resp.status_code == 200:
            raise RuntimeError(f"GET from {base_url+slug} failed with {resp.status_code}, {resp.text}.")

        resource_list = resp.json()
        if progress_bar is not None:
            progress_bar.update(len(resource_list))

        while "next" in resp.links:
            slug = resp.links["next"]["url"]
            resp = self._session.get(base_url + slug)
            if not resp.status_code == 200:
                raise RuntimeError(f"GET from {base_url+slug} failed with {resp.status_code}, {resp.text}.")
            try:
                this_resource_list = resp.json()
            except JSONDecodeError as e:
                print(f"Failed to decode json from {base_url+slug}")
                raise e
            resource_list.extend(this_resource_list)
            if progress_bar is not None:
                progress_bar.update(len(this_resource_list))

        return resource_list

    def _get_base_url(self) -> Tuple[str, str]:
        if ":5000" in self._endpoint:
            # Running locally
            parts = self._endpoint.split(":5000")
            base_url, slug = parts
            base_url += ":5000"
        else:
            parts = self._endpoint.split(".com")
            base_url, slug = parts
            base_url += ".com"
        return base_url, slug


def get_session_that_retries() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session

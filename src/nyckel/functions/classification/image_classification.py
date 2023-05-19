import base64
import os
import time
from io import BytesIO
from typing import Dict, List, Tuple, Union

import requests
from PIL import Image

from nyckel.auth import OAuth2Renewer
from nyckel.functions.classification.classification import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationFunctionHandler,
    ClassificationFunctionURLHandler,
    ClassificationLabel,
    ClassificationLabelHandler,
    ClassificationPrediction,
    ImageClassificationSample,
)
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, get_session_that_retries, repeated_get


class ImageClassificationFunction(ClassificationFunction):
    # TODO: allow users to toggle recoding and resizing.
    # TODO: enable sending the original bytes of no resizing desired.
    # TODO: enable chunking of create and list samples endpoints.

    def __init__(self, function_id: str, auth: OAuth2Renewer, target_largest_side: int = 1024) -> None:
        self._function_id = function_id
        self._auth = auth
        self.target_largest_side = target_largest_side
        self._function_handler = ClassificationFunctionHandler(function_id, auth)
        self._label_handler = ClassificationLabelHandler(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()

        self._function_handler.validate_function()

    @classmethod
    def create_function(cls, name: str, auth: OAuth2Renewer) -> "ImageClassificationFunction":
        return ImageClassificationFunction(ClassificationFunctionHandler.create_function(name, "Text", auth), auth)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        metrics = self.metrics
        status_string = f"[{self._url_handler.train_page}] sampleCount: {metrics['sampleCount']}, annotatedSampleCount: {metrics['annotatedSampleCount']}, predictionCount: {metrics['predictionCount']}, labelCount: {len(metrics['annotatedLabelCounts'])},"
        return status_string

    def __call__(self, sample_data: str) -> ClassificationPrediction:
        self._refresh_auth_token()
        image: Image.Image = ImageDecoder()(sample_data)
        body = {"data": self._to_inline_data(image)}
        endpoint = self._url_handler.api_endpoint("invoke")
        response = self._session.post(endpoint, json=body)

        if "No model available" in response.text:
            raise ValueError(
                f"No model trained yet for this function. Go to {self._url_handler.train_page} to see function status."
            )

        assert response.status_code == 200, f"Invoke failed with {response.status_code=}, {response.text=}"

        return ClassificationPrediction(
            label_name=response.json()["labelName"],
            confidence=response.json()["confidence"],
        )

    @property
    def metrics(self) -> Dict:
        return self._function_handler.get_metrics()

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        if len(sample_data_list) > 500:
            raise ValueError("Too many images. Please chunk up the list.")
        self._refresh_auth_token()
        if not self.has_trained_model():
            raise ValueError(
                f"No model trained yet for this function. Go to {self._url_handler.train_page} to see function status."
            )

        print(f"Invoking {len(sample_data_list)} samples to {self._url_handler.train_page} ...")

        bodies = [{"data": self._to_inline_data(ImageDecoder()(sample_data))} for sample_data in sample_data_list]
        endpoint = self._url_handler.api_endpoint("invoke")
        response_list = ParallelPoster(self._session, endpoint)(bodies)
        return [
            ClassificationPrediction(
                label_name=response.json()["labelName"],
                confidence=response.json()["confidence"],
            )
            for response in response_list
        ]

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        return self._label_handler.create_labels(labels)

    def list_labels(self) -> List[ClassificationLabel]:
        return self._label_handler.list_labels()

    def read_label(self, label_id: str) -> ClassificationLabel:
        return self._label_handler.read_label(label_id)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        return self._label_handler.update_label(label)

    def delete_label(self, label_id: str) -> None:
        return self._label_handler.delete_label(label_id)

    def create_samples(self, samples: List[ImageClassificationSample]) -> List[str]:  # type: ignore
        self._refresh_auth_token()
        print(f"Posting {len(samples)} samples to {self._url_handler.train_page} ...")

        bodies = []
        decoder = ImageDecoder()
        for sample in samples:
            body: Dict[str, Union[str, Dict, None]] = {
                "data": self._to_inline_data(decoder(sample.data)),
                "externalId": sample.external_id,
            }
            if sample.annotation:
                body["annotation"] = {"labelName": sample.annotation.label_name}
            bodies.append(body)

        endpoint = self._url_handler.api_endpoint("samples")
        responses = ParallelPoster(self._session, endpoint)(bodies)
        return [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

    def list_samples(self) -> List[ImageClassificationSample]:  # type: ignore
        self._refresh_auth_token()
        samples_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("samples"))
        samples_typed = [self._sample_from_dict(entry) for entry in samples_dict_list]
        return samples_typed

    def read_sample(self, sample_id: str) -> ImageClassificationSample:
        self._refresh_auth_token()
        response = self._session.get(self._url_handler.api_endpoint(f"samples/{sample_id}"))
        if response.status_code == 404:
            # If calling read right after create, the resource might not be available. Sleep and retry once.
            time.sleep(1)
            response = self._session.get(self._url_handler.api_endpoint(f"samples/{sample_id}"))
        if not response.status_code == 200:
            raise RuntimeError(f"Unable to fetch sample {sample_id} from {self._url_handler.train_page}")
        return self._sample_from_dict(response.json())

    def update_sample(self, sample: ImageClassificationSample) -> ImageClassificationSample:  # type: ignore
        raise NotImplementedError

    def delete_sample(self, sample_id: str) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint(f"samples/{sample_id}")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Sample {sample_id} deleted.")

    def delete(self) -> None:
        self._refresh_auth_token()
        endpoint = self._url_handler.api_endpoint("")
        response = self._session.delete(endpoint)
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"
        print(f"Function {self._url_handler.train_page} deleted.")

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def _to_inline_data(self, img: Image.Image) -> str:
        return ImageEncoder().to_base64(self._resize_image(img))

    def _resize_image(self, img: Image.Image) -> Image.Image:
        def get_new_width_height(width: int, height: int) -> Tuple[int, int]:
            if width < self.target_largest_side and height < self.target_largest_side:
                return width, height
            if width > height:
                new_width = self.target_largest_side
                new_height = int(new_width * height / width)
            else:
                new_height = self.target_largest_side
                new_width = int(new_height * width / height)
            return new_width, new_height

        width, height = get_new_width_height(img.width, img.height)
        img = img.resize((width, height))
        return img

    def _sample_from_dict(self, sample_dict: Dict) -> ImageClassificationSample:
        if "annotation" in sample_dict:
            annotation = ClassificationAnnotation(
                label_name=self._label_handler.get_label_name(sample_dict["annotation"]["labelId"]),
            )
        else:
            annotation = None
        if "prediction" in sample_dict:
            prediction = ClassificationPrediction(
                confidence=sample_dict["prediction"]["confidence"],
                label_name=self._label_handler.get_label_name(sample_dict["prediction"]["labelId"]),
            )
        else:
            prediction = None
        return ImageClassificationSample(
            id=strip_nyckel_prefix(sample_dict["id"]),
            data=sample_dict["data"],
            external_id=sample_dict["externalId"] if "externalId" in sample_dict else None,
            annotation=annotation,
            prediction=prediction,
        )


class ImageDecoder:
    def __call__(self, sample_data: str) -> Image.Image:
        if self._looks_like_url(sample_data):
            return self._load_from_url(sample_data)
        if self._looks_like_local_filepath(sample_data):
            return Image.open(sample_data)
        return self._load_from_data_uri(sample_data)

    def _looks_like_url(self, sample_data: str) -> bool:
        return sample_data.startswith("https://") or sample_data.startswith("http://")

    def _load_from_url(self, url: str) -> Image.Image:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def _looks_like_local_filepath(self, local_path: str) -> bool:
        return os.path.exists(local_path)

    def _load_from_data_uri(self, data_uri: str) -> Image.Image:
        self._validate_image_data_uri(data_uri)
        image_b64_encoded_string = self.strip_base64_prefix(data_uri)
        im_bytes = base64.b64decode(image_b64_encoded_string)
        im_file = BytesIO(im_bytes)
        try:
            img = Image.open(im_file)
        except OSError:
            raise ValueError("Truncated Image Bytes")
        return img

    def _validate_image_data_uri(self, inline_data: str) -> None:
        if inline_data == "":
            raise ValueError("Empty string")
        if "base64" not in inline_data:
            raise ValueError("base64 not in preamble.")
        inline_data_parts = inline_data.split(";base64,")
        if not len(inline_data_parts) == 2:
            raise ValueError("Unable to parse byte string.")
        if inline_data_parts[1] == "":
            raise ValueError("Empty image content")

    def strip_base64_prefix(self, inline_data: str) -> str:
        return inline_data.split(";base64,")[1]


class ImageEncoder:
    def to_base64(self, img: Image.Image) -> str:
        buffered = BytesIO()
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img.save(buffered, format="JPEG", quality=95)
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return "data:image/jpg;base64," + encoded_string

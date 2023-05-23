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
from nyckel.utils import chunkify_list


class ImageClassificationFunction(ClassificationFunction):
    def __init__(
        self, function_id: str, auth: OAuth2Renewer, target_largest_side: int = 1024, force_recode: bool = True
    ) -> None:
        self._function_id = function_id
        self._auth = auth
        self._target_largest_side = target_largest_side
        self._force_recode = force_recode  # Whether to resize & recode images before uploading them.
        self._function_handler = ClassificationFunctionHandler(function_id, auth)
        self._label_handler = ClassificationLabelHandler(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._decoder = ImageDecoder()
        self._encoder = ImageEncoder()
        self._session = get_session_that_retries()

        self._function_handler.validate_function()

    @classmethod
    def create_function(cls, name: str, auth: OAuth2Renewer) -> "ImageClassificationFunction":
        return ImageClassificationFunction(ClassificationFunctionHandler.create_function(name, "Image", auth), auth)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        metrics = self.metrics
        status_string = f"[{self._url_handler.train_page}] sampleCount: {metrics['sampleCount']}, annotatedSampleCount: {metrics['annotatedSampleCount']}, predictionCount: {metrics['predictionCount']}, labelCount: {len(metrics['annotatedLabelCounts'])},"
        return status_string

    def __call__(self, sample_data: str) -> ClassificationPrediction:
        self._refresh_auth_token()
        body = {"data": self._sample_data_to_body(sample_data)}
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

    def invoke(self, sample_data_list: List[str]) -> List[ClassificationPrediction]:
        self._refresh_auth_token()
        if not self.has_trained_model():
            raise ValueError(
                f"No model trained yet for this function. Go to {self._url_handler.train_page} to see function status."
            )
        print(f"Invoking {len(sample_data_list)} samples to {self._url_handler.train_page} ...")

        def _invoke_chunk(sample_data_list_chunk: List[str]) -> List[ClassificationPrediction]:
            bodies = [{"data": self._sample_data_to_body(sample_data)} for sample_data in sample_data_list_chunk]
            endpoint = self._url_handler.api_endpoint("invoke")
            response_list = ParallelPoster(self._session, endpoint)(bodies)
            return [
                ClassificationPrediction(
                    label_name=response.json()["labelName"],
                    confidence=response.json()["confidence"],
                )
                for response in response_list
            ]

        predictions: List[ClassificationPrediction] = list()
        for chunk in chunkify_list(sample_data_list, 500):
            predictions.extend(_invoke_chunk(chunk))

        return predictions

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

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

        def _post_chunk(samples_chunk: List[ImageClassificationSample]) -> List[str]:
            # Chunking up the processing since we fetch & resize images before posting.
            bodies = []
            for sample in samples_chunk:
                body: Dict[str, Union[str, Dict, None]] = {
                    "data": self._sample_data_to_body(sample.data),
                    "externalId": sample.external_id,
                }
                if sample.annotation:
                    body["annotation"] = {"labelName": sample.annotation.label_name}
                bodies.append(body)

            endpoint = self._url_handler.api_endpoint("samples")
            responses = ParallelPoster(self._session, endpoint)(bodies)
            sample_ids_chunk = [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]
            return sample_ids_chunk

        sample_ids: List[str] = list()
        for samples_chunk in chunkify_list(samples, 500):
            sample_ids.extend(_post_chunk(samples_chunk))

        return sample_ids

    def list_samples(self) -> List[ImageClassificationSample]:  # type: ignore
        self._refresh_auth_token()
        samples_dict_list = repeated_get(self._session, self._url_handler.api_endpoint("samples"))
        labels = self._label_handler.list_labels()
        label_name_by_id = {label.id: label.name for label in labels}
        samples_typed = [self._sample_from_dict(entry, label_name_by_id) for entry in samples_dict_list]
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
        labels = self._label_handler.list_labels()
        label_name_by_id = {label.id: label.name for label in labels}
        return self._sample_from_dict(response.json(), label_name_by_id)

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

    def _resize_image(self, img: Image.Image) -> Image.Image:
        def get_new_width_height(width: int, height: int) -> Tuple[int, int]:
            if width < self._target_largest_side and height < self._target_largest_side:
                return width, height
            if width > height:
                new_width = self._target_largest_side
                new_height = int(new_width * height / width)
            else:
                new_height = self._target_largest_side
                new_width = int(new_height * width / height)
            return new_width, new_height

        width, height = get_new_width_height(img.width, img.height)
        img = img.resize((width, height))
        return img

    def _sample_data_to_body(self, sample_data: str) -> str:
        """Encodes the sample data as a URL or dataURI as appropriate."""
        if self._is_nyckel_owned_url(sample_data):
            # If the input points to a Nyckel S3 bucket, we know that the image is processed and verified. So just point back to that URL.
            # This overrides the force_resize input and ensures that the image doesn't get recoded twice.
            return sample_data
        else:
            if self._decoder.looks_like_url(sample_data):
                if self._force_recode:
                    return self._encoder.image_to_base64(self._resize_image(self._decoder.to_image(sample_data)))
                else:
                    return sample_data
            if self._decoder.looks_like_local_filepath(sample_data):
                if self._force_recode:
                    return self._encoder.image_to_base64(self._resize_image(self._decoder.to_image(sample_data)))
                else:
                    return self._encoder.stream_to_base64(self._decoder.to_stream(sample_data))
            if self._decoder.looks_like_data_uri(sample_data):
                if self._force_recode:
                    return self._encoder.image_to_base64(self._resize_image(self._decoder.to_image(sample_data)))
                else:
                    return sample_data
            raise ValueError(f"Can't parse input sample.data={sample_data}")

    def _is_nyckel_owned_url(self, sample_data: str) -> bool:
        return sample_data.startswith("https://s3.us-west-2.amazonaws.com/nyckel.server.")

    def _sample_from_dict(self, sample_dict: Dict, label_name_by_id: Dict) -> ImageClassificationSample:
        if "annotation" in sample_dict:
            annotation = ClassificationAnnotation(
                label_name=label_name_by_id[strip_nyckel_prefix(sample_dict["annotation"]["labelId"])],
            )
        else:
            annotation = None
        if "prediction" in sample_dict:
            prediction = ClassificationPrediction(
                confidence=sample_dict["prediction"]["confidence"],
                label_name=label_name_by_id[strip_nyckel_prefix(sample_dict["prediction"]["labelId"])],
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
    def to_image(self, sample_data: str) -> Image.Image:
        byte_stream = self.to_stream(sample_data)
        try:
            img = Image.open(byte_stream)
        except OSError:
            raise ValueError("Truncated Image Bytes")
        return img

    def to_stream(self, sample_data: str) -> BytesIO:
        if self.looks_like_url(sample_data):
            return self._load_from_url(sample_data)
        if self.looks_like_local_filepath(sample_data):
            return self._load_from_local_filepath(sample_data)
        if self.looks_like_data_uri(sample_data):
            return self._load_from_data_uri(sample_data)
        raise ValueError(f"Unable to parse input {sample_data=}.")

    def looks_like_url(self, sample_data: str) -> bool:
        return sample_data.startswith("https://") or sample_data.startswith("http://")

    def _load_from_url(self, url: str) -> BytesIO:
        response = requests.get(url)
        return BytesIO(response.content)

    def looks_like_local_filepath(self, local_path: str) -> bool:
        return os.path.exists(local_path)

    def _load_from_local_filepath(self, local_path: str) -> BytesIO:
        with open(local_path, "rb") as fh:
            return BytesIO(fh.read())

    def looks_like_data_uri(self, data_uri: str) -> bool:
        return data_uri.startswith("data:image")

    def _load_from_data_uri(self, data_uri: str) -> BytesIO:
        self._validate_image_data_uri(data_uri)
        image_b64_encoded_string = self.strip_base64_prefix(data_uri)
        im_bytes = base64.b64decode(image_b64_encoded_string)
        return BytesIO(im_bytes)

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
    def image_to_base64(self, img: Image.Image) -> str:
        buffered = BytesIO()
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img.save(buffered, format="JPEG", quality=95)
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return "data:image/jpg;base64," + encoded_string

    def stream_to_base64(self, im_bytes: BytesIO) -> str:
        encoded_string = base64.b64encode(im_bytes.getvalue()).decode("utf-8")
        return "data:image/jpg;base64," + encoded_string

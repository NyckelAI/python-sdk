import base64
import os
from io import BytesIO
from typing import Dict, List, Sequence, Tuple, Union

import requests
from PIL import Image

from nyckel import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationLabel,
    ClassificationPrediction,
    Credentials,
    ImageClassificationSample,
    ImageSampleData,
    LabelName,
    NyckelId,
)
from nyckel.functions.classification import factory
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.classification.sample_handler import ClassificationSampleHandler
from nyckel.functions.utils import strip_nyckel_prefix


class ImageClassificationFunction(ClassificationFunction):
    """
    Example:

    ```py
    from nyckel import Credentials, ImageClassificationFunction

    credentials = Credentials(client_id="...", client_secret="...")

    func = ImageClassificationFunction.create("IsCatOrDog", credentials)
    func.create_samples([
        ("cat1.jpg", "cat"),
        ("cat2.jpg", "cat"),
        ("dog1.jpg", "dog"),
        ("dog2.jpg", "dog")
    ])

    predictions = func.invoke(["cat_or_dog.jpg"])
    ```
    """

    def __init__(self, function_id: str, credentials: Credentials) -> None:
        self._function_id = function_id
        self._force_recode: bool  # Whether to resize & recode images before uploading them.
        self._target_largest_side: int  # If recoding, what should be the target longest side
        self._function_handler = ClassificationFunctionHandler(function_id, credentials)
        self._label_handler = ClassificationLabelHandler(function_id, credentials)
        self._url_handler = ClassificationFunctionURLHandler(function_id, credentials.server_url)
        self._sample_handler = ClassificationSampleHandler(function_id, credentials)
        self._decoder = ImageDecoder()
        self._encoder = ImageEncoder()
        assert self._function_handler.get_input_modality() == "Image"

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        status_string = f"Name: {self.name}, id: {self.function_id}, url: {self._url_handler.train_page}"
        return status_string

    @property
    def function_id(self) -> NyckelId:
        return self._function_id

    @property
    def sample_count(self) -> int:
        return self._function_handler.sample_count

    @property
    def label_count(self) -> int:
        return self._function_handler.label_count

    @property
    def name(self) -> str:
        return self._function_handler.get_name()

    @classmethod
    def create(cls, name: str, credentials: Credentials) -> "ImageClassificationFunction":
        return factory.ClassificationFunctionFactory.create(name, "Image", credentials)  # type: ignore

    def delete(self) -> None:
        self._function_handler.delete()

    def invoke(  # type: ignore
        self,
        sample_data_list: List[ImageSampleData],
        model_id: str = "",
        force_recode: bool = False,
        target_largest_side: int = 1000,
    ) -> List[ClassificationPrediction]:
        self._force_recode = force_recode
        self._target_largest_side = target_largest_side
        return self._sample_handler.invoke(sample_data_list, self._sample_data_to_body, model_id=model_id)

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

    def create_labels(self, labels: Sequence[Union[ClassificationLabel, str]]) -> List[NyckelId]:
        typed_labels = [
            label if isinstance(label, ClassificationLabel) else ClassificationLabel(name=label) for label in labels
        ]
        return self._label_handler.create_labels(typed_labels)

    def list_labels(self) -> List[ClassificationLabel]:
        return self._label_handler.list_labels(self.label_count)

    def read_label(self, label_id: NyckelId) -> ClassificationLabel:
        return self._label_handler.read_label(label_id)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        return self._label_handler.update_label(label)

    def delete_labels(self, label_ids: List[NyckelId]) -> None:
        return self._label_handler.delete_labels(label_ids)

    def create_samples(  # type: ignore
        self,
        samples: Sequence[  # type: ignore
            Union[
                ImageClassificationSample,
                Tuple[Image.Image, LabelName],
                Image.Image,
                Tuple[ImageSampleData, LabelName],
                ImageSampleData,
            ]
        ],
        force_recode: bool = False,
        target_largest_side: int = 1000,
    ) -> List[NyckelId]:
        self._force_recode = force_recode
        self._target_largest_side = target_largest_side

        typed_samples = self._wrangle_post_samples_input(samples)
        typed_samples = self._strip_label_names(typed_samples)
        self._create_labels_as_needed(typed_samples)

        return self._sample_handler.create_samples(typed_samples, self._sample_data_to_body)

    def list_samples(self) -> List[ImageClassificationSample]:  # type: ignore
        samples_dict_list = self._sample_handler.list_samples(self.sample_count)
        labels = self._label_handler.list_labels(None)

        label_name_by_id = {label.id: label.name for label in labels}

        return [self._sample_from_dict(entry, label_name_by_id) for entry in samples_dict_list]

    def read_sample(self, sample_id: NyckelId) -> ImageClassificationSample:
        sample_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels(None)
        label_name_by_id = {label.id: label.name for label in labels}

        return self._sample_from_dict(sample_dict, label_name_by_id)

    def update_annotation(self, sample: ImageClassificationSample) -> None:  # type: ignore
        self._sample_handler.update_annotation(sample)

    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        self._sample_handler.delete_samples(sample_ids)

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
            # If the input points to a Nyckel S3 bucket, we know that the image is processed and verified.
            # In that case, we just point back to that URL.
            # This overrides the force_resize input and ensures that the image doesn't get recoded twice.
            return sample_data

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

    def _wrangle_post_samples_input(
        self,
        samples: Sequence[
            Union[
                ImageSampleData,
                Image.Image,
                ImageClassificationSample,
                Tuple[ImageSampleData, LabelName],
                Tuple[Image.Image, LabelName],
            ]
        ],
    ) -> List[ImageClassificationSample]:
        typed_samples: List[ImageClassificationSample] = []
        for sample in samples:
            if isinstance(sample, str):
                typed_samples.append(ImageClassificationSample(data=sample))
            elif isinstance(sample, Image.Image):
                typed_samples.append(ImageClassificationSample(data=self._encoder.image_to_base64(sample)))
            elif isinstance(sample, (tuple, list)) and isinstance(sample[0], str):
                image_str, label_name = sample
                typed_samples.append(
                    ImageClassificationSample(
                        data=image_str, annotation=ClassificationAnnotation(label_name=label_name)
                    )
                )
            elif isinstance(sample, (tuple, list)) and isinstance(sample[0], Image.Image):
                image_pil, label_name = sample
                typed_samples.append(
                    ImageClassificationSample(
                        data=self._encoder.image_to_base64(image_pil),
                        annotation=ClassificationAnnotation(label_name=label_name),
                    )
                )
            elif isinstance(sample, ImageClassificationSample):
                typed_samples.append(sample)
            else:
                raise ValueError(f"Unknown sample type: {type(sample)}")
        return typed_samples

    def _create_labels_as_needed(self, samples: List[ImageClassificationSample]) -> None:
        existing_labels = self._label_handler.list_labels(None)
        existing_label_names = {label.name for label in existing_labels}
        new_label_names = {sample.annotation.label_name for sample in samples if sample.annotation}
        missing_label_names = new_label_names - existing_label_names
        missing_labels = [ClassificationLabel(name=label_name) for label_name in missing_label_names]
        if len(missing_labels) > 0:
            self._label_handler.create_labels(missing_labels)

    def _strip_label_names(self, samples: List[ImageClassificationSample]) -> List[ImageClassificationSample]:
        for sample in samples:
            if sample.annotation:
                sample.annotation.label_name = sample.annotation.label_name.strip()
        return samples


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
        response = requests.get(url, timeout=5)
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

import abc
from typing import Dict, List, Sequence, Union

from PIL import Image

from nyckel import (
    ClassificationLabel,
    ClassificationPrediction,
    Credentials,
    ImageEncoder,
    ImageSampleData,
    ImageTagsSample,
    NyckelId,
    TagsAnnotation,
    TagsPrediction,
)
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.tags import tags_function_factory
from nyckel.functions.tags.tags import TagsFunctionURLHandler
from nyckel.functions.tags.tags_function_handler import TagsFunctionHandler
from nyckel.functions.tags.tags_sample_handler import TagsSampleHandler
from nyckel.functions.utils import ImageSampleBodyTransformer, strip_nyckel_prefix


class ImageTagsFunctionInterface(abc.ABC):

    @abc.abstractmethod
    def __init__(self, function_id: str, credentials: Credentials):
        pass

    @property
    @abc.abstractmethod
    def function_id(self) -> NyckelId:
        pass

    @property
    @abc.abstractmethod
    def sample_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def label_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def create(cls, name: str, credentials: Credentials) -> "ImageTagsFunction":
        pass

    @abc.abstractmethod
    def delete(self) -> None:
        pass

    @abc.abstractmethod
    def invoke(self, sample_data_list: List[ImageSampleData]) -> List[TagsPrediction]:
        """Invokes the trained function. Raises ValueError if function is not trained"""
        pass

    @abc.abstractmethod
    def has_trained_model(self) -> bool:
        pass

    @abc.abstractmethod
    def create_labels(self, labels: Sequence[Union[ClassificationLabel, str]]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_labels(self) -> List[ClassificationLabel]:
        pass

    @abc.abstractmethod
    def read_label(self, label_id: NyckelId) -> ClassificationLabel:
        pass

    @abc.abstractmethod
    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        pass

    @abc.abstractmethod
    def delete_labels(self, label_ids: List[NyckelId]) -> None:
        pass

    @abc.abstractmethod
    def create_samples(self, samples: Sequence[Union[ImageTagsSample, ImageSampleData, Image.Image]]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_samples(self) -> List[ImageTagsSample]:
        pass

    @abc.abstractmethod
    def read_sample(self, sample_id: NyckelId) -> ImageTagsSample:
        pass

    @abc.abstractmethod
    def update_annotation(self, sample: ImageTagsSample) -> None:
        pass

    @abc.abstractmethod
    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        pass


class ImageTagsFunction(ImageTagsFunctionInterface):
    """
    Example:

    ```py
    from nyckel import Credentials, ImageTagsFunction, ImageTagsSample, TagsAnnotation

    credentials = Credentials(client_id="...", client_secret="...")

    func = ImageTagsFunction.create("ClothingColor", credentials)
    func.create_samples([
        ImageTagsSample(data="t-shirt1.jpg", annotation=[TagsAnnotation("White"), TagsAnnotation("Blue")]),
        ImageTagsSample(data="t=shirt2.jpg", annotation=[TagsAnnotation("Red"), TagsAnnotation("White")]),
        ImageTagsSample(data="jacket.jpg", annotation=[TagsAnnotation("Black")]),
        ImageTagsSample(data="jeans.jpg", annotation=[TagsAnnotation("Blue")]),
    ])

    predictions = func.invoke(["new-jacket.jpg"])
    ```
    """

    def __init__(self, function_id: NyckelId, credentials: Credentials):
        self._function_id = function_id

        self._function_handler = TagsFunctionHandler(function_id, credentials)
        self._label_handler = ClassificationLabelHandler(function_id, credentials)
        self._url_handler = TagsFunctionURLHandler(function_id, credentials.server_url)
        self._sample_handler = TagsSampleHandler(function_id, credentials)
        self._encoder = ImageEncoder()

        assert self._function_handler.get_input_modality() == "Image"

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
    def create(cls, name: str, credentials: Credentials) -> "ImageTagsFunction":
        return tags_function_factory.TagsFunctionFactory().create(name, "Image", credentials)  # type:ignore

    def delete(self) -> None:
        self._function_handler.delete()

    def invoke(self, sample_data_list: List[ImageSampleData]) -> List[TagsPrediction]:
        return self._sample_handler.invoke(sample_data_list, ImageSampleBodyTransformer())  # type: ignore

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

    def create_labels(self, labels: Sequence[Union[ClassificationLabel, str]]) -> List[NyckelId]:  # type:ignore
        typed_labels = [
            label if isinstance(label, ClassificationLabel) else ClassificationLabel(name=label)  # type:ignore
            for label in labels
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

    def create_samples(self, samples: Sequence[Union[ImageTagsSample, ImageSampleData, Image.Image]]) -> List[NyckelId]:
        typed_samples = self._wrangle_post_samples_input(samples)
        typed_samples = self._strip_label_names(typed_samples)
        self._create_labels_as_needed(typed_samples)
        return self._sample_handler.create_samples(typed_samples, ImageSampleBodyTransformer())

    def _wrangle_post_samples_input(
        self, samples: Sequence[Union[ImageTagsSample, ImageSampleData]]
    ) -> List[ImageTagsSample]:
        typed_samples: List[ImageTagsSample] = []
        for sample in samples:
            if isinstance(sample, str):
                typed_samples.append(ImageTagsSample(data=sample))
            elif isinstance(sample, Image.Image):
                typed_samples.append(ImageTagsSample(data=self._encoder.to_base64(sample)))
            elif isinstance(sample, ImageTagsSample):
                typed_samples.append(sample)
            else:
                raise ValueError(f"Unknown sample type: {type(sample)}")
        return typed_samples

    def _strip_label_names(self, samples: List[ImageTagsSample]) -> List[ImageTagsSample]:
        for sample in samples:
            if sample.annotation:
                for entry in sample.annotation:
                    entry.label_name = entry.label_name.strip()
        return samples

    def _create_labels_as_needed(self, samples: List[ImageTagsSample]) -> None:
        existing_labels = self._label_handler.list_labels(None)
        existing_label_names = {label.name for label in existing_labels}
        new_label_names: set = set()
        for sample in samples:
            if sample.annotation:
                new_label_names |= {annotation.label_name for annotation in sample.annotation}
        missing_label_names = new_label_names - existing_label_names
        missing_labels = [ClassificationLabel(name=label_name) for label_name in missing_label_names]
        if len(missing_labels) > 0:
            self._label_handler.create_labels(missing_labels)

    def list_samples(self) -> List[ImageTagsSample]:
        samples_dict_list = self._sample_handler.list_samples(self.sample_count)
        labels = self._label_handler.list_labels(None)
        label_name_by_id = {label.id: label.name for label in labels}

        return [self._sample_from_dict(entry, label_name_by_id) for entry in samples_dict_list]  # type: ignore

    def _sample_from_dict(self, sample_dict: Dict, label_name_by_id: Dict) -> ImageTagsSample:
        if "annotation" in sample_dict:
            annotation = [
                TagsAnnotation(
                    label_name=label_name_by_id[strip_nyckel_prefix(entry["labelId"])],
                    present=entry["present"],
                )
                for entry in sample_dict["annotation"]
            ]
        else:
            annotation = None

        if "prediction" in sample_dict:
            prediction = [
                ClassificationPrediction(
                    confidence=entry["confidence"],
                    label_name=label_name_by_id[strip_nyckel_prefix(entry["labelId"])],
                )
                for entry in sample_dict["prediction"]
            ]
        else:
            prediction = None

        return ImageTagsSample(
            id=strip_nyckel_prefix(sample_dict["id"]),
            data=sample_dict["data"],
            external_id=sample_dict["externalId"] if "externalId" in sample_dict else None,
            annotation=annotation,
            prediction=prediction,
        )

    def read_sample(self, sample_id: NyckelId) -> ImageTagsSample:
        sample_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels(None)
        label_name_by_id = {strip_nyckel_prefix(label.id): label.name for label in labels}  # type: ignore

        return self._sample_from_dict(sample_dict, label_name_by_id)  # type: ignore

    def update_annotation(self, sample: ImageTagsSample) -> None:
        self._sample_handler.update_annotation(sample)

    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        self._sample_handler.delete_samples(sample_ids)

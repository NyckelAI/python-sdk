import abc
from typing import Callable, Dict, List, Sequence, Union

from nyckel import (
    ClassificationLabel,
    ClassificationPrediction,
    Credentials,
    NyckelId,
    TabularFunctionField,
    TabularSampleData,
    TabularTagsSample,
    TagsAnnotation,
    TagsPrediction,
)
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.classification.tabular_classification import TabularFieldHandler
from nyckel.functions.tags import tags_function_factory
from nyckel.functions.tags.tags import TagsFunctionURLHandler
from nyckel.functions.tags.tags_function_handler import TagsFunctionHandler
from nyckel.functions.tags.tags_sample_handler import TagsSampleHandler
from nyckel.functions.utils import ImageFieldTransformer, strip_nyckel_prefix


class TabularTagsFunctionInterface(abc.ABC):

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
    def create(cls, name: str, credentials: Credentials) -> "TabularTagsFunction":
        pass

    @abc.abstractmethod
    def delete(self) -> None:
        pass

    @abc.abstractmethod
    def invoke(self, sample_data_list: List[TabularSampleData]) -> List[TagsPrediction]:
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
    def create_fields(self, fields: List[TabularFunctionField]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_fields(self) -> List[TabularFunctionField]:
        pass

    @abc.abstractmethod
    def read_field(self, field_id: NyckelId) -> TabularFunctionField:
        pass

    @abc.abstractmethod
    def delete_field(self, field_id: NyckelId) -> None:
        pass

    @abc.abstractmethod
    def create_samples(self, samples: Sequence[Union[TabularTagsSample, TabularSampleData]]) -> List[NyckelId]:
        pass

    @abc.abstractmethod
    def list_samples(self) -> List[TabularTagsSample]:
        pass

    @abc.abstractmethod
    def read_sample(self, sample_id: NyckelId) -> TabularTagsSample:
        pass

    @abc.abstractmethod
    def update_annotation(self, sample: TabularTagsSample) -> None:
        pass

    @abc.abstractmethod
    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        pass


class TabularTagsFunction(TabularTagsFunctionInterface):
    """
    Example:

    ```py

    from nyckel import Credentials, TabularTagsFunction, TabularTagsSample, TagsAnnotation, TabularFunctionField

    credentials = Credentials(client_id="...", client_secret="...")

    func = TabularTagsFunction.create("NewsTopics", credentials)
    func.create_fields([
        TabularFunctionField(type="Text", name="Title"),
        TabularFunctionField(type="Text", name="Abstract")
    ])
    func.create_samples([
        TabularTagsSample(data={"Title": "New restaurant in SOHO", "Abstract": "This is the best..."}, annotation=[TagsAnnotation("Food"), TagsAnnotation("Reviews")]),
        TabularTagsSample(data={"Title": "Belly-up still going strong", "Abstract": "The Belly-Up tavern in Solana..."}, annotation=[TagsAnnotation("Music"), TagsAnnotation("Reviews")]),
        TabularTagsSample(data={"Title": "Carbonara at its best", "Abstract": "Here is how to make the best..."}, annotation=[TagsAnnotation("Food")]),
        TabularTagsSample(data={"Title": "New album out!", "Abstract": "Taylor swift just released ..."}, annotation=[TagsAnnotation("Music")]),
    ])

    predictions = func.invoke([{"Title": "Swedish meatballs: the best recipe", "Abstract": "This age-old Swedish classic ..."}])
    ```
    """

    def __init__(self, function_id: NyckelId, credentials: Credentials):
        self._function_id = function_id

        self._function_handler = TagsFunctionHandler(function_id, credentials)
        self._label_handler = ClassificationLabelHandler(function_id, credentials)
        self._url_handler = TagsFunctionURLHandler(function_id, credentials.server_url)
        self._sample_handler = TagsSampleHandler(function_id, credentials)
        self._field_handler = TabularFieldHandler(function_id, credentials)

        assert self._function_handler.get_input_modality() == "Tabular"

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
    def create(cls, name: str, credentials: Credentials) -> "TabularTagsFunction":
        return tags_function_factory.TagsFunctionFactory().create(name, "Tabular", credentials)  # type:ignore

    def delete(self) -> None:
        self._function_handler.delete()

    def invoke(self, sample_data_list: List[TabularSampleData]) -> List[TagsPrediction]:
        return self._sample_handler.invoke(sample_data_list, self._get_image_field_transformer())

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
        self._label_handler.delete_labels(label_ids)

    def create_fields(self, fields: List[TabularFunctionField]) -> List[NyckelId]:
        return self._field_handler.create_fields(fields)

    def list_fields(self) -> List[TabularFunctionField]:
        return self._field_handler.list_fields()

    def read_field(self, field_id: NyckelId) -> TabularFunctionField:
        return self._field_handler.read_field(field_id)

    def delete_field(self, field_id: NyckelId) -> None:
        return self._field_handler.delete_field(field_id)

    def create_samples(self, samples: Sequence[Union[TabularTagsSample, TabularSampleData]]) -> List[NyckelId]:
        if len(samples) == 0:
            return []

        typed_samples = self._wrangle_post_samples_input(samples)
        typed_samples = self._strip_label_names(typed_samples)
        self._assert_fields_created(typed_samples)
        self._create_labels_as_needed(typed_samples)
        return self._sample_handler.create_samples(typed_samples, self._get_image_field_transformer())

    def _wrangle_post_samples_input(
        self, samples: Sequence[Union[TabularTagsSample, TabularSampleData]]
    ) -> List[TabularTagsSample]:
        typed_samples: List[TabularTagsSample] = []
        for sample in samples:
            if isinstance(sample, TabularTagsSample):
                typed_samples.append(sample)
            elif isinstance(sample, dict):
                typed_samples.append(TabularTagsSample(data=sample))
            else:
                raise ValueError(f"Sample {sample} has invalid type: {type(sample)}")
        return typed_samples

    def _strip_label_names(self, samples: List[TabularTagsSample]) -> List[TabularTagsSample]:
        for sample in samples:
            if sample.annotation:
                for entry in sample.annotation:
                    entry.label_name = entry.label_name.strip()
        return samples

    def _assert_fields_created(self, samples: List[TabularTagsSample]) -> None:
        existing_fields = self.list_fields()
        existing_field_names = {field.name for field in existing_fields}
        new_field_names = {field_name for sample in samples for field_name in sample.data.keys()}
        missing_field_names = new_field_names - existing_field_names
        assert len(missing_field_names) == 0, f"Fields not created: {missing_field_names=}. Please create fields first."

    def _create_labels_as_needed(self, samples: List[TabularTagsSample]) -> None:
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

    def _get_image_field_transformer(self) -> Callable:
        fields = self.list_fields()
        image_field_transformer = lambda x: x  # noqa: E731
        for field in fields:
            if field.type == "Image":
                # There is only one image field (max) per function, so we can break here.
                image_field_transformer = ImageFieldTransformer(field.name)
                break
        return image_field_transformer

    def list_samples(self) -> List[TabularTagsSample]:
        samples_dict_list = self._sample_handler.list_samples(self.sample_count)
        labels = self._label_handler.list_labels(None)
        fields = self.list_fields()

        label_name_by_id = {label.id: label.name for label in labels}
        field_name_by_id = {field.id: field.name for field in fields}  # type: ignore

        return [self._sample_from_dict(entry, label_name_by_id, field_name_by_id) for entry in samples_dict_list]  # type: ignore # noqa: E501

    def _sample_from_dict(
        self, sample_dict: Dict, label_name_by_id: Dict[str, str], field_name_by_id: Dict[str, str]
    ) -> TabularTagsSample:

        tabular_data_body = {
            field_name_by_id[strip_nyckel_prefix(field_id)]: field_data
            for field_id, field_data in sample_dict["data"].items()
        }

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
            # TODO: Note that we filter out predictsion that are not in the label list.
            # This is a temporary fix since these should not be there in the first place.

            prediction = [
                ClassificationPrediction(
                    confidence=entry["confidence"],
                    label_name=label_name_by_id[strip_nyckel_prefix(entry["labelId"])],
                )
                for entry in sample_dict["prediction"]
                if strip_nyckel_prefix(entry["labelId"]) in label_name_by_id
            ]
        else:
            prediction = None

        return TabularTagsSample(
            id=strip_nyckel_prefix(sample_dict["id"]),
            data=tabular_data_body,
            external_id=sample_dict["externalId"] if "externalId" in sample_dict else None,
            annotation=annotation,
            prediction=prediction,
        )

    def read_sample(self, sample_id: NyckelId) -> TabularTagsSample:
        sample_as_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels(None)
        fields = self.list_fields()

        label_name_by_id = {label.id: label.name for label in labels}
        field_name_by_id = {field.id: field.name for field in fields}

        return self._sample_from_dict(sample_as_dict, label_name_by_id, field_name_by_id)

    def update_annotation(self, sample: TabularTagsSample) -> None:
        self._sample_handler.update_annotation(sample)

    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        self._sample_handler.delete_samples(sample_ids)

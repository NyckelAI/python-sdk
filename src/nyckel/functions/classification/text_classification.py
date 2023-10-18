from typing import Dict, List, Tuple, Union

from nyckel import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationLabel,
    ClassificationPrediction,
    LabelName,
    NyckelId,
    TextClassificationSample,
    TextSampleData,
    User,
)
from nyckel.functions.classification import factory
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.classification.sample_handler import ClassificationSampleHandler
from nyckel.functions.utils import strip_nyckel_prefix


class TextClassificationFunction(ClassificationFunction):
    """
    Example:

    ```py

    from nyckel import User, TextClassificationFunction

    user = User(client_id="...", client_secret="...")

    func = TextClassificationFunction.new("IsToxic", user)
    func.create_samples([
        ("This is a nice comment", "Not toxic"),
        ("Hello friend", "Not toxic"),
        ("This is a bad comment", "Toxic"),
        ("Who is this? Go away!", "Toxic"),
    ])

    prediction = func("This example is fantastic!")
    ```
    """

    def __init__(self, function_id: NyckelId, user: User):
        self._function_id = function_id
        self._user = user

        self._function_handler = ClassificationFunctionHandler(function_id, user)
        self._label_handler = ClassificationLabelHandler(function_id, user)
        self._url_handler = ClassificationFunctionURLHandler(function_id, user.server_url)
        self._sample_handler = ClassificationSampleHandler(function_id, user)

        assert self._function_handler.get_input_modality() == "Text"

    @classmethod
    def new(cls, name: str, user: User) -> "TextClassificationFunction":
        return factory.ClassificationFunctionFactory.new(name, "Text", user)  # type:ignore

    @classmethod
    def create_function(cls, name: str, user: User) -> "TextClassificationFunction":
        return cls.new(name, user)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        status_string = f"Name: {self.name}, id: {self.function_id}, url: {self.train_page}"
        return status_string

    def __call__(self, sample_data: TextSampleData) -> ClassificationPrediction:
        return self.invoke([sample_data])[0]

    @property
    def function_id(self) -> str:
        return self._function_id

    @property
    def sample_count(self) -> int:
        return self._function_handler.sample_count

    @property
    def label_count(self) -> int:
        return self._function_handler.label_count

    @property
    def train_page(self) -> str:
        return self._url_handler.train_page

    @property
    def input_modality(self) -> str:
        return "Text"

    @property
    def name(self) -> str:
        return self._function_handler.get_name()

    @property
    def metrics(self) -> Dict:
        return self._function_handler.get_metrics()

    def invoke(self, sample_data_list: List[TextSampleData]) -> List[ClassificationPrediction]:
        return self._sample_handler.invoke(sample_data_list, lambda x: x)

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

    def create_labels(self, labels: List[ClassificationLabel]) -> List[NyckelId]:
        return self._label_handler.create_labels(labels)

    def list_labels(self) -> List[ClassificationLabel]:
        return self._label_handler.list_labels(self.label_count)

    def read_label(self, label_id: NyckelId) -> ClassificationLabel:
        return self._label_handler.read_label(label_id)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        return self._label_handler.update_label(label)

    def delete_label(self, label_id: NyckelId) -> None:
        return self._label_handler.delete_label(label_id)

    def delete_labels(self, label_ids: List[NyckelId]) -> None:
        return self._label_handler.delete_labels(label_ids)

    def create_samples(
        self, samples: List[Union[TextClassificationSample, Tuple[TextSampleData, LabelName], TextSampleData]]  # type: ignore  # noqa: E501
    ) -> List[NyckelId]:
        typed_samples = self._wrangle_post_samples_input(samples)
        self._create_labels_as_needed(typed_samples)

        return self._sample_handler.create_samples(typed_samples, lambda x: x)

    def list_samples(self) -> List[TextClassificationSample]:  # type: ignore
        samples_dict_list = self._sample_handler.list_samples(self.sample_count)
        labels = self._label_handler.list_labels(None)

        label_name_by_id = {label.id: label.name for label in labels}

        return [self._sample_from_dict(entry, label_name_by_id) for entry in samples_dict_list]  # type: ignore

    def read_sample(self, sample_id: NyckelId) -> TextClassificationSample:
        sample_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels(None)
        label_name_by_id = {strip_nyckel_prefix(label.id): label.name for label in labels}  # type: ignore

        return self._sample_from_dict(sample_dict, label_name_by_id)  # type: ignore

    def update_annotation(self, sample: TextClassificationSample) -> None:  # type: ignore
        self._sample_handler.update_annotation(sample)

    def delete_sample(self, sample_id: NyckelId) -> None:
        self._sample_handler.delete_sample(sample_id)

    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        self._sample_handler.delete_samples(sample_ids)

    def delete(self) -> None:
        self._function_handler.delete()

    def _sample_from_dict(self, sample_dict: Dict, label_name_by_id: Dict[NyckelId, str]) -> TextClassificationSample:
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
        return TextClassificationSample(
            id=strip_nyckel_prefix(sample_dict["id"]),
            data=sample_dict["data"],
            external_id=sample_dict["externalId"] if "externalId" in sample_dict else None,
            annotation=annotation,
            prediction=prediction,
        )

    def _wrangle_post_samples_input(
        self, samples: List[Union[TextClassificationSample, Tuple[TextSampleData, LabelName], TextSampleData]]
    ) -> List[TextClassificationSample]:
        typed_samples: List[TextClassificationSample] = []
        for sample in samples:
            if isinstance(sample, str):
                typed_samples.append(TextClassificationSample(data=sample))
            elif isinstance(sample, tuple):
                typed_samples.append(
                    TextClassificationSample(data=sample[0], annotation=ClassificationAnnotation(label_name=sample[1]))
                )
            elif isinstance(sample, TextClassificationSample):
                typed_samples.append(sample)
            else:
                raise ValueError(f"Unknown sample type: {type(sample)}")
        return typed_samples

    def _create_labels_as_needed(self, samples: List[TextClassificationSample]) -> None:
        existing_labels = self._label_handler.list_labels(None)
        existing_label_names = {label.name for label in existing_labels}
        new_label_names = {sample.annotation.label_name for sample in samples if sample.annotation}
        missing_label_names = new_label_names - existing_label_names
        missing_labels = [ClassificationLabel(name=label_name) for label_name in missing_label_names]
        if len(missing_labels) > 0:
            self._label_handler.create_labels(missing_labels)

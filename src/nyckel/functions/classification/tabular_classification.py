import time
from typing import Callable, Dict, List, Sequence, Tuple, Union

from tqdm import tqdm

from nyckel import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationLabel,
    ClassificationPrediction,
    Credentials,
    LabelName,
    NyckelId,
    TabularClassificationSample,
    TabularFunctionField,
    TabularSampleData,
)
from nyckel.functions.classification import factory
from nyckel.functions.classification.classification import ClassificationFunctionURLHandler
from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.classification.sample_handler import ClassificationSampleHandler
from nyckel.functions.utils import ImageFieldTransformer, strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, SequentialGetter


class TabularClassificationFunction(ClassificationFunction):
    """
    Example:

    ```py

    from nyckel import Credentials, TabularClassificationFunction, TabularFunctionField

    credentials = Credentials(client_id="...", client_secret="...")

    func = TabularClassificationFunction.create("InterestedProspect", credentials)
    func.create_fields([
        TabularFunctionField(type="Text", name="Name"),
        TabularFunctionField(type="Text", name="Response")
    ])
    func.create_samples([
        ({"Name": "Adam Adams", "Response": "Thanks for reaching out. I'd love to chat"}, "Interested"),
        ({"Name": "Bo Berg", "Response": "Sure! Can you tell me a bit more?"}, "Interested"),
        ({"Name": "Charles Carter", "Response": "No thanks, I don't need a Classification API"}, "Not Interested"),
        ({"Name": "Devin Duncan", "Response": "Nope. Please stop bugging me."}, "Not Interested"),
    ])

    predictions = func.invoke([{"Name": "Frank Fisher", "Response": "Yea, I'd love to try the Nyckel API!"}])
    ```
    """

    def __init__(self, function_id: NyckelId, credentials: Credentials) -> None:
        self._function_id = function_id

        self._function_handler = ClassificationFunctionHandler(function_id, credentials)
        self._label_handler = ClassificationLabelHandler(function_id, credentials)
        self._field_handler = TabularFieldHandler(function_id, credentials)
        self._sample_handler = ClassificationSampleHandler(function_id, credentials)
        self._url_handler = ClassificationFunctionURLHandler(function_id, credentials.server_url)
        assert self._function_handler.get_input_modality() == "Tabular"

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        status_string = f"Name: {self.name}, id: {self.function_id}, url: {self._url_handler.train_page}"
        return status_string

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
    def name(self) -> str:
        return self._function_handler.get_name()

    @classmethod
    def create(cls, name: str, credentials: Credentials) -> "TabularClassificationFunction":
        return factory.ClassificationFunctionFactory.create(name, "Tabular", credentials)  # type: ignore

    def delete(self) -> None:
        self._function_handler.delete()

    def invoke(  # type: ignore
        self,
        sample_data_list: List[TabularSampleData],
        model_id: str = "",
    ) -> List[ClassificationPrediction]:
        return self._sample_handler.invoke(sample_data_list, self._get_image_field_transformer(), model_id=model_id)

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

    def create_fields(self, fields: List[TabularFunctionField]) -> List[NyckelId]:
        return self._field_handler.create_fields(fields)

    def list_fields(self) -> List[TabularFunctionField]:
        return self._field_handler.list_fields()

    def read_field(self, field_id: NyckelId) -> TabularFunctionField:
        return self._field_handler.read_field(field_id)

    def delete_field(self, field_id: NyckelId) -> None:
        return self._field_handler.delete_field(field_id)

    def create_samples(self, samples: Sequence[Union[TabularClassificationSample, Tuple[TabularSampleData, LabelName], TabularSampleData]]) -> List[NyckelId]:  # type: ignore # noqa: E501
        if len(samples) == 0:
            return []

        typed_samples = self._wrangle_post_samples_input(samples)
        typed_samples = self._strip_label_names(typed_samples)
        self._assert_fields_created(typed_samples)
        self._create_labels_as_needed(typed_samples)
        return self._sample_handler.create_samples(typed_samples, self._get_image_field_transformer())

    def _get_image_field_transformer(self) -> Callable:
        fields = self.list_fields()
        image_field_transformer = lambda x: x  # noqa: E731
        for field in fields:
            if field.type == "Image":
                # There is only one image field (max) per function, so we can break here.
                image_field_transformer = ImageFieldTransformer(field.name)
                break
        return image_field_transformer

    def list_samples(self) -> List[TabularClassificationSample]:  # type: ignore
        samples_dict_list = self._sample_handler.list_samples(self.sample_count)
        labels = self._label_handler.list_labels(None)
        fields = self.list_fields()

        label_name_by_id = {label.id: label.name for label in labels}
        field_name_by_id = {field.id: field.name for field in fields}  # type: ignore

        return [self._sample_from_dict(entry, label_name_by_id, field_name_by_id) for entry in samples_dict_list]  # type: ignore # noqa: E501

    def read_sample(self, sample_id: NyckelId) -> TabularClassificationSample:
        sample_as_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels(None)
        fields = self.list_fields()

        label_name_by_id = {label.id: label.name for label in labels}
        field_name_by_id = {field.id: field.name for field in fields}  # type: ignore

        return self._sample_from_dict(sample_as_dict, label_name_by_id, field_name_by_id)  # type: ignore

    def update_annotation(self, sample: TabularClassificationSample) -> None:  # type: ignore
        self._sample_handler.update_annotation(sample)

    def delete_samples(self, sample_ids: List[NyckelId]) -> None:
        self._sample_handler.delete_samples(sample_ids)

    def _wrangle_post_samples_input(
        self,
        samples: Sequence[Union[TabularClassificationSample, Tuple[TabularSampleData, LabelName], TabularSampleData]],
    ) -> List[TabularClassificationSample]:
        typed_samples: List[TabularClassificationSample] = []
        for sample in samples:
            if isinstance(sample, TabularClassificationSample):
                typed_samples.append(sample)
            elif isinstance(sample, (list, tuple)):
                data_dict, label_name = sample
                typed_samples.append(
                    TabularClassificationSample(
                        data=data_dict, annotation=ClassificationAnnotation(label_name=label_name)
                    )
                )
            elif isinstance(sample, dict):
                typed_samples.append(TabularClassificationSample(data=sample))
            else:
                raise ValueError(f"Unknown sample type: {type(sample)}")
        return typed_samples

    def _assert_fields_created(self, samples: List[TabularClassificationSample]) -> None:
        existing_fields = self.list_fields()
        existing_field_names = {field.name for field in existing_fields}
        new_field_names = {field_name for sample in samples for field_name in sample.data.keys()}
        missing_field_names = new_field_names - existing_field_names
        assert len(missing_field_names) == 0, f"Fields not created: {missing_field_names=}. Please create fields first."

    def _create_labels_as_needed(self, samples: List[TabularClassificationSample]) -> None:
        existing_labels = self._label_handler.list_labels(None)
        existing_label_names = {label.name for label in existing_labels}
        new_label_names = {sample.annotation.label_name for sample in samples if sample.annotation}
        missing_label_names = new_label_names - existing_label_names
        missing_labels = [ClassificationLabel(name=label_name) for label_name in missing_label_names]
        if len(missing_labels) > 0:
            self._label_handler.create_labels(missing_labels)

    def _sample_from_dict(
        self, sample_dict: Dict, label_name_by_id: Dict[str, str], field_name_by_id: Dict[str, str]
    ) -> TabularClassificationSample:
        tabular_data_body = {
            field_name_by_id[strip_nyckel_prefix(field_id)]: field_data
            for field_id, field_data in sample_dict["data"].items()
        }

        if "externalId" in sample_dict:
            external_id = sample_dict["externalId"]
        else:
            external_id = None

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

        return TabularClassificationSample(
            id=strip_nyckel_prefix(sample_dict["id"]),
            data=tabular_data_body,
            external_id=external_id,
            annotation=annotation,
            prediction=prediction,
        )

    def _strip_label_names(self, samples: List[TabularClassificationSample]) -> List[TabularClassificationSample]:
        for sample in samples:
            if sample.annotation:
                sample.annotation.label_name = sample.annotation.label_name.strip()
        return samples


class TabularFieldHandler:
    def __init__(self, function_id: NyckelId, credentials: Credentials):
        self._function_id = function_id
        self._credentials = credentials
        self._url_handler = ClassificationFunctionURLHandler(function_id, credentials.server_url)
        self._field_name_by_id: Dict = {}

    def create_fields(self, fields: List[TabularFunctionField]) -> List[NyckelId]:
        bodies = [{"name": field.name, "type": field.type} for field in fields]
        url = self._url_handler.api_endpoint(path="fields")
        session = self._credentials.get_session()
        progress_bar = tqdm(total=len(bodies), ncols=80, desc="Posting fields")
        responses = ParallelPoster(session, url, progress_bar)(bodies)
        field_ids = [strip_nyckel_prefix(resp.json()["id"]) for resp in responses]

        self._confirm_new_fields_available(fields)
        return field_ids

    def _confirm_new_fields_available(self, new_fields: List[TabularFunctionField]) -> None:
        # Before returning, make sure the assets are available via the API.
        new_fields_available_via_api = False
        timeout_seconds = 5
        t0 = time.time()
        while not new_fields_available_via_api:
            if time.time() - t0 > timeout_seconds:
                raise ValueError("Something went wrong when posting fields.")
            time.sleep(0.5)
            fields_retrieved = self.list_fields()
            new_fields_available_via_api = set([label.name for label in new_fields]).issubset(
                [label.name for label in fields_retrieved]
            )

    def list_fields(self) -> List[TabularFunctionField]:
        session = self._credentials.get_session()
        fields_dict_list = SequentialGetter(session, self._url_handler.api_endpoint(path="fields"))(
            tqdm(ncols=80, desc="Listing fields")
        )
        return [self._field_from_dict(entry) for entry in fields_dict_list]

    def read_field(self, field_id: NyckelId) -> TabularFunctionField:
        session = self._credentials.get_session()
        url = self._url_handler.api_endpoint(path=f"fields/{field_id}")
        response = session.get(url)
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch field {field_id} from {self._url_handler.train_page} "
                f"{response.text=} {response.status_code=}"
            )
        return self._field_from_dict(response.json())

    def delete_field(self, field_id: NyckelId) -> None:
        session = self._credentials.get_session()
        response = session.delete(self._url_handler.api_endpoint(path=f"fields/{field_id}"))
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"

    def _field_from_dict(self, field_dict: Dict) -> TabularFunctionField:
        return TabularFunctionField(
            name=field_dict["name"], id=strip_nyckel_prefix(field_dict["id"]), type=field_dict["type"]
        )

import numbers
import time
from typing import Dict, List

from tqdm import tqdm

from nyckel import OAuth2Renewer
from nyckel.functions.classification import factory
from nyckel.functions.classification.classification import (
    ClassificationAnnotation,
    ClassificationFunction,
    ClassificationFunctionURLHandler,
    ClassificationLabel,
    ClassificationPrediction,
    TabularClassificationSample,
    TabularFunctionField,
)
from nyckel.functions.classification.function_handler import ClassificationFunctionHandler
from nyckel.functions.classification.label_handler import ClassificationLabelHandler
from nyckel.functions.classification.sample_handler import ClassificationSampleHandler
from nyckel.functions.utils import strip_nyckel_prefix
from nyckel.request_utils import ParallelPoster, SequentialGetter, get_session_that_retries


class TabularClassificationFunction(ClassificationFunction):
    def __init__(self, function_id: str, auth: OAuth2Renewer) -> None:
        self._function_id = function_id
        self._auth = auth

        self._function_handler = ClassificationFunctionHandler(function_id, auth)
        self._label_handler = ClassificationLabelHandler(function_id, auth)
        self._field_handler = TabularFieldHandler(function_id, auth)
        self._sample_handler = ClassificationSampleHandler(function_id, auth)
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()
        self._refresh_auth_token()
        assert self._function_handler.get_input_modality() == "Tabular"

    @classmethod
    def new(cls, name: str, auth: OAuth2Renewer) -> "TabularClassificationFunction":
        return factory.ClassificationFunctionFactory.new(name, "Tabular", auth)  # type: ignore

    @classmethod
    def create_function(cls, name: str, auth: OAuth2Renewer) -> "TabularClassificationFunction":
        return cls.new(name, auth)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        status_string = f"Name: {self.get_name()}, id: {self.function_id}, url: {self.train_page}"
        return status_string

    def __call__(self, sample_data: Dict) -> ClassificationPrediction:  # type: ignore
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
        return "Tabular"

    def get_name(self) -> str:
        return self._function_handler.get_name()

    @property
    def metrics(self) -> Dict:
        return self._function_handler.get_metrics()

    def invoke(self, sample_data_list: List[Dict]) -> List[ClassificationPrediction]:  # type: ignore
        return self._sample_handler.invoke(sample_data_list, lambda x: x)

    def has_trained_model(self) -> bool:
        return self._function_handler.is_trained

    def create_labels(self, labels: List[ClassificationLabel]) -> List[str]:
        return self._label_handler.create_labels(labels)

    def list_labels(self) -> List[ClassificationLabel]:
        return self._label_handler.list_labels(self.label_count)

    def read_label(self, label_id: str) -> ClassificationLabel:
        return self._label_handler.read_label(label_id)

    def update_label(self, label: ClassificationLabel) -> ClassificationLabel:
        return self._label_handler.update_label(label)

    def delete_label(self, label_id: str) -> None:
        return self._label_handler.delete_label(label_id)

    def delete_labels(self, label_ids: List[str]) -> None:
        return self._label_handler.delete_labels(label_ids)

    def create_samples(self, samples: List[TabularClassificationSample]) -> List[str]:  # type: ignore # noqa: E501
        self._create_fields_as_needed(samples)
        existing_fields = self._field_handler.list_fields()
        field_id_by_name = {field.name: field.id for field in existing_fields}
        samples = [self._switch_field_names_to_field_ids(sample, field_id_by_name) for sample in samples]  # type: ignore # noqa: E501
        return self._sample_handler.create_samples(samples, lambda x: x)

    def _switch_field_names_to_field_ids(
        self, sample: TabularClassificationSample, field_id_by_name: Dict[str, str]
    ) -> TabularClassificationSample:
        field_names = list(sample.data.keys())
        for field_name in field_names:
            field_value = sample.data.pop(field_name)
            sample.data[field_id_by_name[field_name]] = field_value
        return sample

    def list_samples(self) -> List[TabularClassificationSample]:  # type: ignore
        samples_dict_list = self._sample_handler.list_samples(self.sample_count)
        labels = self._label_handler.list_labels(None)
        fields = self._field_handler.list_fields()

        label_name_by_id = {label.id: label.name for label in labels}
        field_name_by_id = {field.id: field.name for field in fields}  # type: ignore

        return [self._sample_from_dict(entry, label_name_by_id, field_name_by_id) for entry in samples_dict_list]  # type: ignore # noqa: E501

    def read_sample(self, sample_id: str) -> TabularClassificationSample:
        sample_as_dict = self._sample_handler.read_sample(sample_id)

        labels = self._label_handler.list_labels(None)
        fields = self._field_handler.list_fields()

        label_name_by_id = {label.id: label.name for label in labels}
        field_name_by_id = {field.id: field.name for field in fields}  # type: ignore

        return self._sample_from_dict(sample_as_dict, label_name_by_id, field_name_by_id)  # type: ignore

    def update_annotation(self, sample: TabularClassificationSample) -> None:  # type: ignore
        self._sample_handler.update_annotation(sample)

    def delete_sample(self, sample_id: str) -> None:
        self._sample_handler.delete_sample(sample_id)

    def delete_samples(self, sample_ids: List[str]) -> None:
        self._sample_handler.delete_samples(sample_ids)

    def delete(self) -> None:
        self._function_handler.delete()

    def _create_fields_as_needed(self, samples: List[TabularClassificationSample]) -> None:
        existing_fields = self._field_handler.list_fields()

        new_fields: List[TabularFunctionField] = []
        for sample in samples:
            for field_name, field_value in sample.data.items():
                if field_name in [field.name for field in existing_fields + new_fields]:
                    continue
                if isinstance(field_value, numbers.Number):
                    new_fields.append(TabularFunctionField(name=field_name, type="Number"))
                else:
                    new_fields.append(TabularFunctionField(name=field_name, type="Text"))

        if len(new_fields) > 0:
            self._field_handler.create_fields(new_fields)

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

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})


class TabularFieldHandler:
    def __init__(self, function_id: str, auth: OAuth2Renewer):
        self._function_id = function_id
        self._auth = auth
        self._url_handler = ClassificationFunctionURLHandler(function_id, auth.server_url)
        self._session = get_session_that_retries()
        self._field_name_by_id: Dict = {}

    def _refresh_auth_token(self) -> None:
        self._session.headers.update({"authorization": "Bearer " + self._auth.token})

    def create_fields(self, fields: List[TabularFunctionField]) -> List[str]:
        self._refresh_auth_token()
        bodies = [{"name": field.name, "type": field.type} for field in fields]
        url = self._url_handler.api_endpoint(path="fields")
        responses = ParallelPoster(self._session, url)(bodies)
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
        self._refresh_auth_token()
        fields_dict_list = SequentialGetter(self._session, self._url_handler.api_endpoint(path="fields"))(
            tqdm(ncols=80, desc="Listing fields")
        )
        return [self._field_from_dict(entry) for entry in fields_dict_list]

    def read_field(self, field_id: str) -> TabularFunctionField:
        self._refresh_auth_token()
        url = self._url_handler.api_endpoint(path=f"fields/{field_id}")
        response = self._session.get(url)
        if not response.status_code == 200:
            raise RuntimeError(
                f"Unable to fetch field {field_id} from {self._url_handler.train_page} "
                f"{response.text=} {response.status_code=}"
            )
        return self._field_from_dict(response.json())

    def delete_field(self, field_id: str) -> None:
        self._refresh_auth_token()
        response = self._session.delete(self._url_handler.api_endpoint(path=f"fields/{field_id}"))
        assert response.status_code == 200, f"Delete failed with {response.status_code=}, {response.text=}"

    def _field_from_dict(self, field_dict: Dict) -> TabularFunctionField:
        return TabularFunctionField(
            name=field_dict["name"], id=strip_nyckel_prefix(field_dict["id"]), type=field_dict["type"]
        )

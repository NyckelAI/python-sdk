import time
import warnings
from typing import Dict, Tuple, Union

import pytest
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    TabularClassificationFunction,
    TabularClassificationSample,
)


def test_end_to_end(tabular_classification_function: TabularClassificationFunction) -> None:
    func: TabularClassificationFunction = tabular_classification_function

    func.create_labels([ClassificationLabel(name="happy"), ClassificationLabel(name="sad")])

    func.create_samples(
        [
            TabularClassificationSample(
                data={"firstname": "Adam", "lastname": "Art"}, annotation=ClassificationAnnotation(label_name="happy")
            ),
            TabularClassificationSample(
                data={"firstname": "Bo", "lastname": "Busy"}, annotation=ClassificationAnnotation(label_name="happy")
            ),
            TabularClassificationSample(
                data={"firstname": "Carl", "lastname": "Carrot"}, annotation=ClassificationAnnotation(label_name="sad")
            ),
            TabularClassificationSample(
                data={"firstname": "Dick", "lastname": "Denali"}, annotation=ClassificationAnnotation(label_name="sad")
            ),
        ]
    )

    fields = func._field_handler.list_fields()
    assert len(fields) == 2
    assert set([field.name for field in fields]) == set(["firstname", "lastname"])
    time.sleep(2)
    assert len(func.list_samples()) == 4
    assert len(func.list_labels()) == 2

    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    prediction = func.invoke([{"firstname": "Eric", "lastname": "Ebony"}])[0]
    assert isinstance(prediction, ClassificationPrediction)

    predictions = func.invoke(
        [{"firstname": "Eric", "lastname": "Ebony"}, {"firstname": "Frank", "lastname": "Froggy"}]
    )
    assert isinstance(predictions[0], ClassificationPrediction)


def test_field_creation(tabular_classification_function: TabularClassificationFunction) -> None:
    func: TabularClassificationFunction = tabular_classification_function

    func.create_samples([TabularClassificationSample(data={"firstname": "Adam"})])
    fields = func._field_handler.list_fields()
    assert len(fields) == 1
    assert fields[0].type == "Text"
    func.create_samples([TabularClassificationSample(data={"age": 32})])
    fields = func._field_handler.list_fields()
    assert len(fields) == 2
    age_field = [field for field in fields if field.name == "age"][0]
    assert age_field.type == "Number"

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Can't assign a Text value to a Number field. This sample will not be created.
    assert len(func.create_samples([TabularClassificationSample(data={"age": "Twelve"})])) == 0
    warnings.filterwarnings("default", category=FutureWarning)


post_sample_parameter_examples = [
    TabularClassificationSample(
        data={"firstname": "Adam", "lastname": "Abrams"}, annotation=ClassificationAnnotation(label_name="Person")
    ),
    ({"firstname": "Adam", "lastname": "Abrams"}, "Person"),
    {"firstname": "Adam", "lastname": "Abrams"},
]


@pytest.mark.parametrize("post_samples_input", post_sample_parameter_examples)
def test_post_sample_overloading(
    tabular_classification_function: TabularClassificationFunction,
    post_samples_input: Union[TabularClassificationSample, Tuple[Dict, str], Dict],
) -> None:
    tabular_classification_function.create_samples([post_samples_input])
    time.sleep(1)
    samples = tabular_classification_function.list_samples()
    assert len(samples) == 1
    assert samples[0].data == {"firstname": "Adam", "lastname": "Abrams"}

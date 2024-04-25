import os
import time
import warnings
from typing import Dict, Tuple, Union

import pytest
from conftest import large_image_url, make_random_image, small_image_url
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    ImageDecoder,
    TabularClassificationFunction,
    TabularClassificationSample,
)
from nyckel.functions.classification.classification import TabularFunctionField


def test_end_to_end(tabular_classification_function: TabularClassificationFunction) -> None:
    func: TabularClassificationFunction = tabular_classification_function

    func.create_labels([ClassificationLabel(name="happy"), ClassificationLabel(name="sad")])
    func.create_fields(
        [
            TabularFunctionField(name="firstname", type="Text"),
            TabularFunctionField(name="lastname", type="Text"),
            TabularFunctionField(name="mugshot", type="Image"),
        ]
    )
    local_image_file = os.path.abspath("tests/fixtures/flower.jpg")
    func.create_samples(
        [
            TabularClassificationSample(
                data={"firstname": "Adam", "lastname": "Art", "mugshot": local_image_file},
                annotation=ClassificationAnnotation(label_name="happy"),
            ),
            TabularClassificationSample(
                data={"firstname": "Bo", "lastname": "Busy", "mugshot": local_image_file},
                annotation=ClassificationAnnotation(label_name="happy"),
            ),
            TabularClassificationSample(
                data={"firstname": "Carl", "lastname": "Carrot", "mugshot": local_image_file},
                annotation=ClassificationAnnotation(label_name="sad"),
            ),
            TabularClassificationSample(
                data={"firstname": "Dick", "lastname": "Denali", "mugshot": local_image_file},
                annotation=ClassificationAnnotation(label_name="sad"),
            ),
        ]
    )

    fields = func.list_fields()
    assert len(fields) == 3
    assert set([field.name for field in fields]) == set(["firstname", "lastname", "mugshot"])
    time.sleep(2)
    assert len(func.list_samples()) == 4
    assert len(func.list_labels()) == 2

    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    prediction = func.invoke([{"firstname": "Eric", "lastname": "Ebony", "mugshot": local_image_file}])[0]
    assert isinstance(prediction, ClassificationPrediction)

    predictions = func.invoke(
        [
            {"firstname": "Eric", "lastname": "Ebony", "mugshot": local_image_file},
            {"firstname": "Frank", "lastname": "Froggy", "mugshot": local_image_file},
        ]
    )
    assert isinstance(predictions[0], ClassificationPrediction)


def test_field_creation(tabular_classification_function: TabularClassificationFunction) -> None:
    func: TabularClassificationFunction = tabular_classification_function
    func.create_fields([TabularFunctionField(name="firstname", type="Text")])
    func.create_samples([TabularClassificationSample(data={"firstname": "Adam"})])
    fields = func.list_fields()
    assert len(fields) == 1
    assert fields[0].type == "Text"
    func.create_fields([TabularFunctionField(name="age", type="Number")])
    func.create_samples([TabularClassificationSample(data={"age": 32})])
    fields = func.list_fields()
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
    tabular_classification_function.create_fields(
        [TabularFunctionField(name="firstname", type="Text"), TabularFunctionField(name="lastname", type="Text")]
    )
    tabular_classification_function.create_samples([post_samples_input])
    time.sleep(1)
    samples = tabular_classification_function.list_samples()
    assert len(samples) == 1
    assert samples[0].data == {"firstname": "Adam", "lastname": "Abrams"}


def test_image_field(tabular_classification_function: TabularClassificationFunction) -> None:
    tabular_classification_function.create_fields(
        [TabularFunctionField(name="firstname", type="Text"), TabularFunctionField(name="mug", type="Image")]
    )

    local_filepath = os.path.abspath("tests/fixtures/flower.jpg")

    tabular_classification_function.create_samples(
        [
            TabularClassificationSample(
                data={"firstname": "Adam", "mug": local_filepath},
                annotation=ClassificationAnnotation(label_name="Person"),
            )
        ]
    )
    time.sleep(1)
    samples = tabular_classification_function.list_samples()
    assert isinstance(samples[0].data["mug"], str)
    img = ImageDecoder().to_image(samples[0].data["mug"])

    # The tabular filed uploader will resize the images to 384x384
    assert max(img.size) == 384


local_image_file = os.path.abspath("tests/fixtures/flower.jpg")


class TestImageFieldOverloading:
    """Testing various way of giving the image field in the sample data."""

    @pytest.mark.parametrize(
        "sample",
        [
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": small_image_url}),
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": large_image_url}),
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": make_random_image()}),
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": local_image_file}),
        ],
    )
    def test_post_sample(
        self,
        tabular_classification_function_with_fields: TabularClassificationFunction,
        sample: TabularClassificationSample,
    ) -> None:
        func = tabular_classification_function_with_fields
        func.create_samples([sample])
        time.sleep(0.5)
        assert func.sample_count == 1


class TestInvokeFieldOverloading:

    @pytest.mark.parametrize(
        "sample",
        [
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": small_image_url}),
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": large_image_url}),
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": make_random_image()}),
            TabularClassificationSample(data={"name": "Adam", "age": 32, "mug": local_image_file}),
        ],
    )
    def test_invoke(
        self,
        trained_tabular_classification_function: TabularClassificationFunction,
        sample: TabularClassificationSample,
    ) -> None:
        func = trained_tabular_classification_function
        pred = func.invoke([sample.data])[0]
        assert isinstance(pred, ClassificationPrediction)

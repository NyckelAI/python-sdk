import os
import time

import pytest
from conftest import large_image_url, make_random_image, small_image_url
from nyckel import ClassificationPrediction, TabularFunctionField, TabularTagsFunction, TabularTagsSample

local_image_file = os.path.abspath("tests/fixtures/flower.jpg")


class TestFields:

    def test_create(self, tabular_tags_function: TabularTagsFunction):
        func = tabular_tags_function
        func.create_fields(
            [
                TabularFunctionField(name="firstname", type="Text"),
                TabularFunctionField(name="height", type="Number"),
                TabularFunctionField(name="picture", type="Image"),
            ]
        )
        fields = func.list_fields()
        assert len(fields) == 3

    def test_read(self, tabular_tags_function: TabularTagsFunction):
        func = tabular_tags_function
        field_id = func.create_fields([TabularFunctionField(name="firstname", type="Text")])[0]
        time.sleep(0.5)

        field = func.read_field(field_id)
        assert field.name == "firstname"
        assert field.type == "Text"

    def test_delete(self, tabular_tags_function: TabularTagsFunction):
        func = tabular_tags_function
        field_id = func.create_fields([TabularFunctionField(name="firstname", type="Text")])[0]
        time.sleep(0.5)

        func.delete_field(field_id)
        time.sleep(0.5)

        fields = func.list_fields()
        assert len(fields) == 0


class TestImageFieldOverloading:
    """Testing various way of giving the image field in the sample data."""

    @pytest.mark.parametrize(
        "sample",
        [
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": small_image_url}),
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": large_image_url}),
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": make_random_image()}),
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": local_image_file}),
        ],
    )
    def test_post_sample(
        self,
        tabular_tags_function_with_fields: TabularTagsFunction,
        sample: TabularTagsSample,
    ) -> None:
        func = tabular_tags_function_with_fields
        func.create_samples([sample])
        time.sleep(0.5)
        assert func.sample_count == 1


class TestInvokeFieldOverloading:

    @pytest.mark.parametrize(
        "sample",
        [
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": small_image_url}),
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": large_image_url}),
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": make_random_image()}),
            TabularTagsSample(data={"name": "Adam", "age": 32, "mug": local_image_file}),
        ],
    )
    def test_invoke(
        self,
        trained_tabular_tags_function: TabularTagsFunction,
        sample: TabularTagsSample,
    ) -> None:
        func = trained_tabular_tags_function
        prediction = func.invoke([sample.data])[0]
        if len(prediction) > 0:
            assert isinstance(prediction[0], ClassificationPrediction)

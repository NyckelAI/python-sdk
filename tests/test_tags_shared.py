""" Shared tests for all Tags functions."""

import time

import pytest
from conftest import make_random_image, make_random_tabular, make_random_text
from nyckel import (
    ClassificationLabel,
    ClassificationPrediction,
    ImageTagsFunction,
    ImageTagsSample,
    TabularFunctionField,
    TabularTagsFunction,
    TabularTagsSample,
    TagsAnnotation,
    TextTagsFunction,
    TextTagsSample,
)


@pytest.mark.parametrize(
    "function_type,function_class",
    [
        ("image_tags_function", ImageTagsFunction),
        ("text_tags_function", TextTagsFunction),
        ("tabular_tags_function", TabularTagsFunction),
    ],
)
class TestFunction:

    # Create & Delete happens in the test fixture, and there is no update operation
    # So remains to test read.

    def test_read(self, function_type, function_class, auth_test_credentials, request):
        func = request.getfixturevalue(function_type)
        function_id = func.function_id
        reloaded_function = function_class(function_id, auth_test_credentials)
        assert reloaded_function.function_id == function_id


@pytest.mark.parametrize(
    "function_class,sample_data_maker",
    [
        (ImageTagsFunction, make_random_image),
        (TextTagsFunction, make_random_text),
        (TabularTagsFunction, make_random_tabular),
    ],
)
class TestProperties:

    def test_all(self, function_class, sample_data_maker, auth_test_credentials):
        name = "PYTHON-SDK TAGS PROPERTIES TEST FUNCTION"
        func = function_class.create(name, auth_test_credentials)
        if isinstance(func, TabularTagsFunction):
            fields = [
                TabularFunctionField(name="name", type="Text"),
                TabularFunctionField(name="age", type="Number"),
                TabularFunctionField(name="mug", type="Image"),
            ]
            func.create_fields(fields)
        assert func.name == name
        assert func.sample_count == 0
        assert func.label_count == 0

        func.create_labels(["cat", "dog"])
        time.sleep(0.5)
        assert func.label_count == 2

        func.create_samples([sample_data_maker(), sample_data_maker()])
        time.sleep(0.5)
        assert func.sample_count == 2

        func.delete()


@pytest.mark.parametrize("function_type", ["image_tags_function", "text_tags_function", "tabular_tags_function"])
class TestLabels:

    def test_create_untyped(self, function_type, request):
        func = request.getfixturevalue(function_type)
        label_ids = func.create_labels(["cat", "dog"])
        assert len(label_ids) == 2

    def test_create_typed(self, function_type, request):
        func = request.getfixturevalue(function_type)
        label_ids = func.create_labels([ClassificationLabel("cat"), ClassificationLabel("dog")])
        assert len(label_ids) == 2

    def test_read(self, function_type, request):
        func = request.getfixturevalue(function_type)
        label_id = func.create_labels(["dog"])[0]
        label = func.read_label(label_id)
        assert label.name == "dog"
        assert label.id == label_id

    def test_update(self, function_type, request):
        func = request.getfixturevalue(function_type)
        label_id = func.create_labels(["dog"])[0]
        updated_label = ClassificationLabel(name="doggo", id=label_id)
        func.update_label(updated_label)
        time.sleep(0.5)
        labels = func.list_labels()
        assert labels[0].name == "doggo"

    def test_delete(self, function_type, request):
        func = request.getfixturevalue(function_type)
        label_id = func.create_labels(["dog"])[0]
        func.delete_labels([label_id])
        time.sleep(0.5)
        labels = func.list_labels()
        assert len(labels) == 0

    def test_strip_whitespaces(self, function_type, request):
        func = request.getfixturevalue(function_type)
        func.create_labels(["  cat  "])
        labels = func.list_labels()
        assert labels[0].name == "cat"


@pytest.mark.parametrize(
    "function_type,sample_data_maker,sample_class",
    [
        ("image_tags_function", make_random_image, ImageTagsSample),
        ("text_tags_function", make_random_text, TextTagsSample),
        ("tabular_tags_function_with_fields", make_random_tabular, TabularTagsSample),
    ],
)
class TestSamples:

    def test_create_untyped(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)
        sample_ids = func.create_samples([sample_data_maker(), sample_data_maker()])
        assert len(sample_ids) == 2

    def test_create_typed(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)
        sample_ids = func.create_samples(
            [sample_class(data=sample_data_maker()), sample_class(data=sample_data_maker())]
        )
        assert len(sample_ids) == 2

    def test_create_annotated(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)
        sample_ids = func.create_samples([sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("Nice")])])
        assert len(sample_ids) == 1
        assert func.label_count == 1  # Labels are created automatically

    def test_read(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)
        sample_to_be_created = sample_data_maker()
        sample_id = func.create_samples([sample_to_be_created])[0]
        time.sleep(0.5)
        sample = func.read_sample(sample_id)
        assert sample.id == sample_id
        if function_type == "text_tags_function":
            # We can't test for exact equality for image and (tabular with image fields) because
            # the image data is recoded by the server.
            assert sample.data == sample_to_be_created

    def test_update(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)
        func.create_labels(["Nice"])
        original_sample = sample_class(data=sample_data_maker())
        sample_id = func.create_samples([original_sample])[0]
        time.sleep(0.5)

        updated_sample = sample_class(id=sample_id, data=original_sample.data, annotation=[TagsAnnotation("Nice")])
        func.update_annotation(updated_sample)
        time.sleep(0.5)

        sample = func.read_sample(sample_id)
        assert sample.annotation[0] == TagsAnnotation("Nice")

    def test_update_with_two_samples(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)

        samples = [
            sample_class(data=sample_data_maker(), annotation=[TagsAnnotation(label_name="Nice", present=True)]),
            sample_class(data=sample_data_maker(), annotation=[TagsAnnotation(label_name="Bad", present=True)]),
        ]

        func.create_samples(samples)
        time.sleep(0.5)

        samples = func.list_samples()  # type: ignore
        samples[0].annotation = [
            TagsAnnotation(label_name="Nice", present=True),
            TagsAnnotation(label_name="Bad", present=True),
        ]
        func.update_annotation(samples[0])
        time.sleep(0.5)

        assert samples[0].id is not None
        updated_sample = func.read_sample(samples[0].id)
        assert updated_sample.annotation == samples[0].annotation

    def test_delete(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)
        sample_id = func.create_samples([sample_data_maker()])[0]
        func.delete_samples([sample_id])
        time.sleep(0.5)
        samples = func.list_samples()
        assert len(samples) == 0

    def test_new_labels(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)

        func.create_labels([ClassificationLabel(name="label1")])
        sample_1_id = func.create_samples(
            [sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("label1")])]
        )[0]

        func.create_labels([ClassificationLabel(name="label2")])

        sample_2_id = func.create_samples(
            [sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("label1")])]
        )[0]
        time.sleep(0.5)

        sample1 = func.read_sample(sample_1_id)

        # When sample1 was created, label2 wasn't present yet.
        # So sample1 doesn't have an "opinion" about that label.
        # This means that it's not present in the annotation field.
        assert len(sample1.annotation) == 1
        assert TagsAnnotation("label1") in sample1.annotation
        assert TagsAnnotation("label2", present=False) not in sample1.annotation

        sample2 = func.read_sample(sample_2_id)
        # When sample2 was created, both labels were present.
        # Since we didn't give an annotation for label2, it was set to not present.
        assert len(sample2.annotation) == 2
        assert TagsAnnotation("label1") in sample2.annotation
        assert TagsAnnotation("label2", present=False) in sample2.annotation


@pytest.mark.parametrize(
    "function_type,sample_data_maker,sample_class",
    [
        ("image_tags_function", make_random_image, ImageTagsSample),
        ("text_tags_function", make_random_text, TextTagsSample),
        ("tabular_tags_function_with_fields", make_random_tabular, TabularTagsSample),
    ],
)
class TestEndToEnd:

    def test_end_to_end(self, function_type, sample_data_maker, sample_class, request):
        func = request.getfixturevalue(function_type)

        labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
        func.create_labels(labels_to_create)

        samples = [
            sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("Nice")]),
            sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("Nice"), TagsAnnotation("Boo")]),
            sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("Boo")]),
            sample_class(data=sample_data_maker(), annotation=[TagsAnnotation("Boo")]),
            sample_class(data=sample_data_maker(), annotation=[]),
        ]
        assert not func.has_trained_model()
        func.create_samples(samples)

        while not func.has_trained_model():
            print("-> No trained model yet. Sleeping 1 sec...")
            time.sleep(1)

        predictions = func.invoke([sample_data_maker(), sample_data_maker(), sample_data_maker()])
        assert len(predictions) == 3
        for prediction in predictions:
            if len(prediction) > 0:
                assert isinstance(prediction[0], ClassificationPrediction)

        returned_samples = func.list_samples()
        assert len(samples) == len(returned_samples)
        assert isinstance(returned_samples[0], sample_class)

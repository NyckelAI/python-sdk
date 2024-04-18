import os
import time

from conftest import make_random_image
from nyckel import ClassificationLabel, ClassificationPrediction, ImageTagsFunction, ImageTagsSample, TagsAnnotation
from PIL import Image


class TestFunction:

    # Create & Delete happens in the test fixture, and there is no update operation
    # So remains to test read.

    def test_read(self, image_tags_function, auth_test_credentials):
        function_id = image_tags_function.function_id
        reloaded_function = ImageTagsFunction(function_id, auth_test_credentials)
        assert reloaded_function.function_id == function_id


class TestLabels:

    def test_create(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        label_ids = func.create_labels(["cat", "dog"])
        assert len(label_ids) == 2

    def test_read(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        label_id = func.create_labels(["dog"])[0]
        label = func.read_label(label_id)
        assert label.name == "dog"
        assert label.id == label_id

    def test_update(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        label_id = func.create_labels(["dog"])[0]
        updated_label = ClassificationLabel(name="doggo", id=label_id)
        func.update_label(updated_label)
        time.sleep(0.5)
        labels = func.list_labels()
        assert labels[0].name == "doggo"

    def test_delete(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        label_id = func.create_labels(["dog"])[0]
        func.delete_labels([label_id])
        time.sleep(0.5)
        labels = func.list_labels()
        assert len(labels) == 0


class TestSamples:

    def test_create_inline(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        sample_ids = func.create_samples([make_random_image(), make_random_image()])
        assert len(sample_ids) == 2

    def test_create_PIL(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        sample_ids = func.create_samples([Image.new("RGB", (100, 100)), Image.new("RGB", (80, 80))])
        assert len(sample_ids) == 2

    def test_create_local_path(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        local_filepath = os.path.abspath("tests/fixtures/flower.jpg")
        sample_ids = func.create_samples([local_filepath])
        assert len(sample_ids) == 1

    def test_create_url(self, image_tags_function: ImageTagsFunction):
        func = image_tags_function
        sample_ids = func.create_samples(["https://picsum.photos/200", "https://picsum.photos/300"])
        assert len(sample_ids) == 2


def test_end_to_end(image_tags_function):
    func: ImageTagsFunction = image_tags_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)

    samples = [
        ImageTagsSample(data=make_random_image(), annotation=[TagsAnnotation("Nice")]),
        ImageTagsSample(data=make_random_image(), annotation=[TagsAnnotation("Nice"), TagsAnnotation("Boo")]),
        ImageTagsSample(data=make_random_image(), annotation=[TagsAnnotation("Boo")]),
        ImageTagsSample(data=make_random_image(), annotation=[TagsAnnotation("Boo")]),
        ImageTagsSample(data=make_random_image(), annotation=[]),
    ]

    func.create_samples(samples)

    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    predictions = func.invoke([make_random_image(), make_random_image(), make_random_image()])
    assert len(predictions) == 3
    for prediction in predictions:
        if len(prediction) > 0:
            assert isinstance(prediction[0], ClassificationPrediction)

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], ImageTagsSample)

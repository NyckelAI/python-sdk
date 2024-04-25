import os

from conftest import large_image_url, make_random_image, small_image_url
from nyckel import ImageTagsFunction
from PIL import Image


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
        sample_ids = func.create_samples([small_image_url, large_image_url])
        assert len(sample_ids) == 2

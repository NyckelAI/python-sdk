import os
import time
from typing import Iterator

from nyckel import ImageClassificationFunction, ImageDecoder

FULLY_TRANSPARENT_PIXEL_THAT_SHOULD_BECOME_WHITE = (10, 10)


def test_white_background(image_classification_function: Iterator[ImageClassificationFunction]):
    func: ImageClassificationFunction = image_classification_function
    local_filepath = os.path.abspath("tests/fixtures/mixed-alpha-background.png")
    sample_id = func.create_samples([local_filepath])[0]
    time.sleep(1)
    sample_back = func.read_sample(sample_id)
    img = ImageDecoder().to_image(sample_back.data)
    assert img.getpixel(FULLY_TRANSPARENT_PIXEL_THAT_SHOULD_BECOME_WHITE) == (255, 255, 255)

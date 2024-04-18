import os
from io import BytesIO

from nyckel import ImageDecoder, ImageEncoder
from PIL import Image

image_url = "https://www.nyckel.com/blog/images/taimi-case-study-header-image.png"

image_base64 = ImageEncoder().to_base64(Image.new(mode="RGB", size=(40, 40)))

image_filepath = os.path.abspath("tests/fixtures/flower.jpg")


def test_to_bytes() -> None:
    decoder = ImageDecoder()

    assert isinstance(decoder.to_stream(image_url), BytesIO)
    assert isinstance(decoder.to_stream(image_base64), BytesIO)
    assert isinstance(decoder.to_stream(image_filepath), BytesIO)


def test_to_image() -> None:
    decoder = ImageDecoder()
    assert isinstance(decoder.to_image(image_url), Image.Image)
    assert isinstance(decoder.to_image(image_base64), Image.Image)
    assert isinstance(decoder.to_image(image_filepath), Image.Image)

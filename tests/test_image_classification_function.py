import os
import time
from typing import Tuple, Union

import numpy as np
import pytest
from conftest import get_test_credentials, make_random_image
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    ImageClassificationFunction,
    ImageClassificationSample,
)
from nyckel.functions.classification.image_classification import ImageDecoder
from PIL import Image

credentials = get_test_credentials()


def test_samples(image_classification_function: ImageClassificationFunction) -> None:
    func: ImageClassificationFunction = image_classification_function
    label = ClassificationLabel(name="Nice")
    func.create_labels([label])
    time.sleep(1)

    images = [make_random_image(), make_random_image()]
    sample = ImageClassificationSample(data=images[0], annotation=ClassificationAnnotation(label_name="Nice"))
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert ImageDecoder().to_image(sample_back.data).size == ImageDecoder().to_image(sample.data).size

    sample = ImageClassificationSample(
        data=images[1], annotation=ClassificationAnnotation(label_name="Nice"), external_id="my_external_id"
    )
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert sample_back.external_id == sample.external_id

    # Check that list_samples work
    samples_back = func.list_samples()
    assert len(samples_back) == 2

    # Check delete
    func.delete_samples(sample_ids[0:1])
    time.sleep(1)
    samples_back = func.list_samples()
    assert len(samples_back) == 1


def test_end_to_end(image_classification_function: ImageClassificationFunction) -> None:
    func: ImageClassificationFunction = image_classification_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)

    samples = [
        ImageClassificationSample(data=make_random_image(), annotation=ClassificationAnnotation(label_name="Nice")),
        ImageClassificationSample(data=make_random_image(), annotation=ClassificationAnnotation(label_name="Nice")),
        ImageClassificationSample(data=make_random_image(), annotation=ClassificationAnnotation(label_name="Boo")),
        ImageClassificationSample(data=make_random_image(), annotation=ClassificationAnnotation(label_name="Boo")),
        ImageClassificationSample(data=make_random_image()),
    ]

    func.create_samples(samples)

    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    assert isinstance(func.invoke([make_random_image()])[0], ClassificationPrediction)

    assert len(func.invoke([make_random_image(), make_random_image(), make_random_image()])) == 3

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], ImageClassificationSample)


@pytest.mark.skipif(
    credentials.server_url == "http://localhost:5000",
    reason="Integrity of image on copy is only implemented for images hosted on S3.",
)
def test_image_integrity_on_copy(image_classification_function: ImageClassificationFunction) -> None:
    func: ImageClassificationFunction = image_classification_function
    local_filepath = os.path.abspath("tests/flower.jpg")

    # When Nyckel first receives an image, it resizes and recodes it.
    first_sample_id = func.create_samples([ImageClassificationSample(data=local_filepath, external_id="v0")])[0]
    first_sample = func.read_sample(first_sample_id)

    # Here we point back to a nyckel owned url, so the images bytes are stored exactly as posted.
    second_sample_id = func.create_samples([ImageClassificationSample(data=first_sample.data, external_id="v1")])[0]
    second_sample = func.read_sample(second_sample_id)

    decoder = ImageDecoder()
    assert np.array_equal(np.array(decoder.to_image(first_sample.data)), np.array(decoder.to_image(second_sample.data)))


post_sample_parameter_examples = [
    ImageClassificationSample(
        data=os.path.abspath("tests/flower.jpg"), annotation=ClassificationAnnotation(label_name="Nice")
    ),
    (os.path.abspath("tests/flower.jpg"), "Nice"),
    (Image.open(os.path.abspath("tests/flower.jpg")), "Nice"),
    os.path.abspath("tests/flower.jpg"),
    Image.open(os.path.abspath("tests/flower.jpg")),
]


@pytest.mark.parametrize("post_samples_input", post_sample_parameter_examples)
def test_post_sample_overloading(
    image_classification_function: ImageClassificationFunction,
    post_samples_input: Union[ImageClassificationSample, Tuple[str, str], Tuple[Image.Image, str], str, Image.Image],
) -> None:
    image_classification_function.create_samples([post_samples_input])
    time.sleep(1)
    samples = image_classification_function.list_samples()
    assert len(samples) == 1

    decoder = ImageDecoder()
    returned_image = decoder.to_image(samples[0].data)
    original_image = Image.open(os.path.abspath("tests/flower.jpg"))

    assert returned_image.size == original_image.size
    # TODO: test that the image content is the same. It's tricky b/c Nyckel re-codes
    # the image when the sample is created.

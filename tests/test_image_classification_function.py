import time
import os
import pytest
import numpy as np
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    ImageClassificationSample,
    ImageClassificationFunction,
)
from nyckel.functions.classification.image_classification import ImageDecoder, ImageEncoder
from PIL import Image
import numpy as np
from conftest import get_test_user

user = get_test_user()


def make_random_image(size=100):
    imarray = np.random.rand(size, size, 3) * 255
    img = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    return ImageEncoder().image_to_base64(img)


def test_samples(image_classification_function):
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
    func.delete_sample(sample_ids[0])
    time.sleep(1)
    samples_back = func.list_samples()
    assert len(samples_back) == 1


def test_end_to_end(image_classification_function):
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
        print("No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    assert isinstance(func(make_random_image()), ClassificationPrediction)

    assert len(func.invoke([make_random_image(), make_random_image(), make_random_image()])) == 3

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], ImageClassificationSample)


@pytest.mark.skipif(
    user.server_url == "http://localhost:5000",
    reason="Integrity of image on copy is only implemented for images hosted on S3.",
)
def test_image_integrity_on_copy(image_classification_function):
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

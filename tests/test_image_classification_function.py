import time
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    ImageClassificationSample,
    ImageClassificationFunction,
)
from io import BytesIO
from PIL import Image
import numpy as np
import base64


def make_random_image(size=100):
    imarray = np.random.rand(size, size, 3) * 255
    img = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    return img


def base64_encode(img: Image.Image, format="JPEG"):
    buffered = BytesIO()
    if not img.mode == "RGB":
        img = img.convert("RGB")
    img.save(buffered, format=format)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/jpg;base64," + encoded_string


def test_samples(auth_test_user):
    func = ImageClassificationFunction.create_function("[TEST IMAGE SAMPLES]", auth_test_user)
    label = ClassificationLabel(name="Nice")
    func.create_labels([label])
    time.sleep(1)

    images = [make_random_image(), make_random_image()]
    sample = ImageClassificationSample(data=images[0], annotation=ClassificationAnnotation(label_name="Nice"))
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert sample_back.data.size == sample.data.size

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
    # assert samples_back[0].data == images[0]

    func.delete()


def test_end_to_end(auth_test_user):
    func = ImageClassificationFunction.create_function("[TEST IMAGE END TO END]", auth_test_user)

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)

    label_names_post_complete = False
    while not label_names_post_complete:
        print("Labels not posted yet. Sleeping 1 sec...")
        time.sleep(1)
        labels = func.list_labels()
        label_names_post_complete = set([l.name for l in labels_to_create]) == set([l.name for l in labels])

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

    func.delete()

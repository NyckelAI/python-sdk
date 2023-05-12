import os
import time

import pytest
from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    OAuth2Renewer,
    TextClassificationFunction,
    TextClassificationSample,
)


@pytest.mark.skipif("NYCKEL_CLIENT_ID" not in os.environ, reason="Env. var NYCKEL_CLIENT_ID not set")
@pytest.mark.skipif("NYCKEL_CLIENT_SECRET" not in os.environ, reason="Env. var NYCKEL_CLIENT_SECRET not set")
def test_samples():
    auth = OAuth2Renewer(
        client_id=os.environ["NYCKEL_CLIENT_ID"],
        client_secret=os.environ["NYCKEL_CLIENT_SECRET"],
        server_url="https://www.nyckel.com",
    )
    func = TextClassificationFunction.create_function("[TEST] labels", auth)
    label = ClassificationLabel(name="Nice")
    func.create_labels([label])

    sample = TextClassificationSample(data="hello", annotation=ClassificationAnnotation(label_name="Nice"))
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert sample_back.data == sample.data

    sample = TextClassificationSample(
        data="hello again", annotation=ClassificationAnnotation(label_name="Nice"), external_id="my_external_id"
    )
    sample_ids = func.create_samples([sample])
    sample_back = func.read_sample(sample_ids[0])
    assert sample_back.data == sample.data
    assert sample_back.external_id == sample.external_id

    # Check that list_samples work
    samples_back = func.list_samples()
    assert len(samples_back) == 2
    assert set([sample.data for sample in samples_back]) == set(["hello", "hello again"])

    # Check delete
    func.delete_sample(sample_ids[0])
    time.sleep(1)
    samples_back = func.list_samples()
    assert len(samples_back) == 1
    assert samples_back[0].data == "hello"

    func.delete()


@pytest.mark.skipif("NYCKEL_CLIENT_ID" not in os.environ, reason="Env. var NYCKEL_CLIENT_ID not set")
@pytest.mark.skipif("NYCKEL_CLIENT_SECRET" not in os.environ, reason="Env. var NYCKEL_CLIENT_SECRET not set")
def test_end_to_end():
    auth = OAuth2Renewer(
        client_id=os.environ["NYCKEL_CLIENT_ID"],
        client_secret=os.environ["NYCKEL_CLIENT_SECRET"],
        server_url="https://www.nyckel.com",
    )

    rate_my_greeting_function = TextClassificationFunction.create_function("RateMyGreeting", auth)

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    rate_my_greeting_function.create_labels(labels_to_create)

    label_names_post_complete = False
    while not label_names_post_complete:
        print("Labels not posted yet. Sleeping 1 sec...")
        time.sleep(1)
        labels = rate_my_greeting_function.list_labels()
        label_names_post_complete = set([l.name for l in labels_to_create]) == set([l.name for l in labels])

    samples = [
        TextClassificationSample(data="hello", annotation=ClassificationAnnotation(label_name="Nice")),
        TextClassificationSample(data="hi", annotation=ClassificationAnnotation(label_name="Nice")),
        TextClassificationSample(data="Good bye", annotation=ClassificationAnnotation(label_name="Boo")),
        TextClassificationSample(data="I'm leaving", annotation=ClassificationAnnotation(label_name="Boo")),
        TextClassificationSample(data="Hi again"),
    ]

    rate_my_greeting_function.create_samples(samples)

    while not rate_my_greeting_function.has_trained_model():
        print("No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    assert isinstance(rate_my_greeting_function("Howdy"), ClassificationPrediction)

    assert len(rate_my_greeting_function.invoke(["Hej", "Hola", "Gruezi"])) == 3

    returned_samples = rate_my_greeting_function.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], TextClassificationSample)

    rate_my_greeting_function.delete()

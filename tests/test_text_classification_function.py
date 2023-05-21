import time

from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    ClassificationPrediction,
    TextClassificationSample,
)


def test_samples(text_classification_function):
    func = text_classification_function
    # func = TextClassificationFunction.create_function("[TEST TEXT CLASSIFICATION SAMPLES]", auth_test_user)
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


def test_end_to_end(text_classification_function):
    func = text_classification_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)

    samples = [
        TextClassificationSample(data="hello", annotation=ClassificationAnnotation(label_name="Nice")),
        TextClassificationSample(data="hi", annotation=ClassificationAnnotation(label_name="Nice")),
        TextClassificationSample(data="Good bye", annotation=ClassificationAnnotation(label_name="Boo")),
        TextClassificationSample(data="I'm leaving", annotation=ClassificationAnnotation(label_name="Boo")),
        TextClassificationSample(data="Hi again"),
    ]

    func.create_samples(samples)

    while not func.has_trained_model():
        print("No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    assert isinstance(func("Howdy"), ClassificationPrediction)

    assert len(func.invoke(["Hej", "Hola", "Gruezi"])) == 3

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], TextClassificationSample)

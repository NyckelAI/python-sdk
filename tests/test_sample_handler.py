import time

from nyckel import ClassificationAnnotation, ClassificationLabel, TextClassificationFunction, TextClassificationSample


def test_simple(text_classification_function_with_content: TextClassificationFunction) -> None:
    func: TextClassificationFunction = text_classification_function_with_content

    samples = func.list_samples()
    assert len(samples) > 0
    func.delete_samples([sample.id for sample in samples if sample.id])
    time.sleep(2)
    samples = func.list_samples()
    assert len(samples) == 0


def test_duplicate_samples(text_classification_function: TextClassificationFunction) -> None:
    func: TextClassificationFunction = text_classification_function

    samples = [
        TextClassificationSample(data="hello"),
        TextClassificationSample(data="hello"),
    ]

    sample_ids = func.create_samples(samples)
    assert len(sample_ids) == 1  # Duplicate samples are ignored when posting.


def test_update_annotation(text_classification_function: TextClassificationFunction) -> None:
    func: TextClassificationFunction = text_classification_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Bad")]
    nice = ClassificationAnnotation(label_name="Nice")
    bad = ClassificationAnnotation(label_name="Bad")
    sample = TextClassificationSample(data="Hello, friend", external_id="1", annotation=nice)

    # Create assets and make sure the annotation is set correctly.
    func.create_labels(labels_to_create)
    sample_id = func.create_samples([sample])[0]
    time.sleep(1)
    sample = func.read_sample(sample_id)
    assert sample.annotation and sample.annotation.label_name == "Nice"

    # Remove the annotation, update it and make sure it's updated.
    sample.annotation = None
    func.update_annotation(sample)
    time.sleep(1)
    sample = func.read_sample(sample_id)
    assert not sample.annotation

    # Change the annotation, update it and make sure it's updated.
    sample.annotation = bad
    func.update_annotation(sample)
    time.sleep(1)
    sample = func.read_sample(sample_id)
    assert sample.annotation and sample.annotation.label_name == "Bad"

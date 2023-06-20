import time

from nyckel import TextClassificationFunction
from nyckel import TextClassificationSample


def test_simple(text_classification_function_with_content):
    func: TextClassificationFunction = text_classification_function_with_content

    samples = func.list_samples()
    assert len(samples) > 0
    func.delete_samples([sample.id for sample in samples])
    time.sleep(2)
    samples = func.list_samples()
    assert len(samples) == 0


def test_duplicate_samples(text_classification_function):
    func: TextClassificationFunction = text_classification_function

    samples = [
        TextClassificationSample(data="hello"),
        TextClassificationSample(data="hello"),
    ]

    sample_ids = func.create_samples(samples)
    assert len(sample_ids) == 1  # Duplicate samples are ignored when posting.

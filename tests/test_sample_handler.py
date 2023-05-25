import time

from nyckel import TextClassificationFunction


def test_simple(text_classification_function_with_content):
    func: TextClassificationFunction = text_classification_function_with_content

    samples = func.list_samples()
    assert len(samples) > 0
    func.delete_samples([sample.id for sample in samples])
    time.sleep(2)
    samples = func.list_samples()
    assert len(samples) == 0

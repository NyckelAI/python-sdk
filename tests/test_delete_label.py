import time

from nyckel import (
    ClassificationAnnotation,
    ClassificationLabel,
    OAuth2Renewer,
    TextClassificationFunction,
    TextClassificationSample,
)
from nyckel.functions.classification.classification import ClassificationFunction
from nyckel.orchestration import NyckelLabelDeleter


def hold_until_list_samples_available(function: ClassificationFunction, expected_count: int) -> None:
    actual_count = 0
    while not actual_count == expected_count:
        actual_count = len(function.list_samples())
        time.sleep(0.25)


def test_simple(text_classification_function: TextClassificationFunction, auth_test_user: OAuth2Renewer) -> None:
    func: TextClassificationFunction = text_classification_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)
    nice = ClassificationAnnotation(label_name="Nice")
    boo = ClassificationAnnotation(label_name="Boo")
    samples = [
        TextClassificationSample(data="hello", external_id="1", annotation=nice),
        TextClassificationSample(data="hi", external_id="2", annotation=nice),
        TextClassificationSample(data="Good bye", external_id="3", annotation=boo),
        TextClassificationSample(data="I'm leaving", external_id="4", annotation=boo),
        TextClassificationSample(data="Hi again", external_id="5"),
    ]
    func.create_samples(samples)
    hold_until_list_samples_available(func, 5)

    deleter = NyckelLabelDeleter(func.function_id, auth_test_user, skip_confirmation=True)
    deleter.delete_label_and_samples("Nice")
    time.sleep(2)

    samples = func.list_samples()
    assert len(samples) == 3
    for sample in samples:
        if sample.annotation:
            assert "Nice" not in sample.annotation.label_name

    labels = func.list_labels()
    assert len(labels) == 1

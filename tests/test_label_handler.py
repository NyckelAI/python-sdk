import time

from nyckel import (
    ClassificationLabel,
    TextClassificationFunction,
)


def test_labels(auth_test_user):
    # Try creating a simple label
    func = TextClassificationFunction.create_function("[TEST LABELS]", auth_test_user)
    label = ClassificationLabel(name="Nice")
    label_ids = func.create_labels([label])
    label_back = func.read_label(label_ids[0])
    assert label_back.name == "Nice"

    # And then one with metadata
    label = ClassificationLabel(name="Nicer", description="Very nice", metadata={"how nice?": "very"})
    label_ids = func.create_labels([label])
    nicer_label_id = label_ids[0]
    label_back = func.read_label(nicer_label_id)
    assert label_back.name == label.name
    assert label_back.description == label.description
    assert label_back.metadata == label.metadata

    # Check that list_labels work
    labels = func.list_labels()
    if len(labels) < 2:
        # Give the API a chance to catch up.
        time.sleep(1)
        labels = func.list_labels()
    assert len(labels) == 2
    assert set([label.name for label in labels]) == set(["Nice", "Nicer"])

    # And delete label
    func.delete_label(nicer_label_id)
    labels = func.list_labels()
    if not len(labels) == 1:
        # Give the API a chance to catch up.
        time.sleep(1)
        labels = func.list_labels()
    assert len(labels) == 1
    assert labels[0].name == "Nice"

    func.delete()


def test_labels_optional_params(auth_test_user):
    # Try creating a simple label
    func = TextClassificationFunction.create_function("[TEST OPTIONAL FIELDS]", auth_test_user)

    label = ClassificationLabel(name="Nicer", description="Very nice")
    label_id = func.create_labels([label])[0]
    label_back = func.read_label(label_id)
    assert label_back.name == label.name
    assert label_back.description == label.description
    assert label_back.metadata == label.metadata

    func.delete()

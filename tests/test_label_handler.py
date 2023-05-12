import os
import time

import pytest
from nyckel import (
    ClassificationLabel,
    OAuth2Renewer,
    TextClassificationFunction,
)


@pytest.mark.skipif("NYCKEL_CLIENT_ID" not in os.environ, reason="Env. var NYCKEL_CLIENT_ID not set")
@pytest.mark.skipif("NYCKEL_CLIENT_SECRET" not in os.environ, reason="Env. var NYCKEL_CLIENT_SECRET not set")
def test_labels():
    auth = OAuth2Renewer(
        client_id=os.environ["NYCKEL_CLIENT_ID"],
        client_secret=os.environ["NYCKEL_CLIENT_SECRET"],
        server_url="https://www.nyckel.com",
    )

    # Try creating a simple label
    func = TextClassificationFunction.create_function("[TEST] labels", auth)
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
    assert len(labels) == 2
    assert set([label.name for label in labels]) == set(["Nice", "Nicer"])

    # And delete label
    func.delete_label(nicer_label_id)
    time.sleep(1)
    labels = func.list_labels()
    assert len(labels) == 1
    assert labels[0].name == "Nice"

    func.delete()


@pytest.mark.skipif("NYCKEL_CLIENT_ID" not in os.environ, reason="Env. var NYCKEL_CLIENT_ID not set")
@pytest.mark.skipif("NYCKEL_CLIENT_SECRET" not in os.environ, reason="Env. var NYCKEL_CLIENT_SECRET not set")
def test_labels_optional_params():
    auth = OAuth2Renewer(
        client_id=os.environ["NYCKEL_CLIENT_ID"],
        client_secret=os.environ["NYCKEL_CLIENT_SECRET"],
        server_url="https://www.nyckel.com",
    )

    # Try creating a simple label
    func = TextClassificationFunction.create_function("[TEST] optional fields", auth)

    label = ClassificationLabel(name="Nicer", description="Very nice")
    label_id = func.create_labels([label])[0]
    label_back = func.read_label(label_id)
    assert label_back.name == label.name
    assert label_back.description == label.description
    assert label_back.metadata == label.metadata

    func.delete()

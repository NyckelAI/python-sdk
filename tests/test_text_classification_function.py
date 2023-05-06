import os
import time

import pytest
from nyckel import OAuth2Renewer, TextClassificationFunction, TextClassificationSample, ClassificationPrediction


@pytest.mark.skipif("NYCKEL_CLIENT_ID" not in os.environ, reason="Env. var NYCKEL_CLIENT_ID not set")
@pytest.mark.skipif("NYCKEL_CLIENT_SECRET" not in os.environ, reason="Env. var NYCKEL_CLIENT_SECRET not set")
def test_end_to_end():
    auth = OAuth2Renewer(
        client_id=os.environ["NYCKEL_CLIENT_ID"],
        client_secret=os.environ["NYCKEL_CLIENT_SECRET"],
        server_url="https://www.nyckel.com",
    )

    samples = [
        TextClassificationSample(data="hello", annotation="Nice"),
        TextClassificationSample(data="hi", annotation="Nice"),
        TextClassificationSample(data="Good bye", annotation="Boo"),
        TextClassificationSample(data="I'm leaving", annotation="Boo"),
        TextClassificationSample(data="Hi again"),
    ]

    rate_my_greeting_function = TextClassificationFunction.new("RateMyGreeting", auth)

    rate_my_greeting_function.add_labels(["Nice", "Boo"])
    rate_my_greeting_function.add_samples(samples)

    has_trained_model = False
    while not has_trained_model:
        time.sleep(1)
        has_trained_model = rate_my_greeting_function.has_trained_model()
        print("No trained model yet. Sleeping 1 sec.")

    assert isinstance(rate_my_greeting_function("Howdy"), ClassificationPrediction)

    assert len(rate_my_greeting_function.invoke(["Hej", "Hola", "Gruezi"])) == 3

    rate_my_greeting_function.delete()

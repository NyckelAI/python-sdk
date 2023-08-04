import json

from nyckel import ImageClassificationFunction, OAuth2Renewer, TabularClassificationFunction, TextClassificationFunction
from nyckel.orchestration import NyckelFunctionDuplicator


def test_copy_image_function(
    image_classification_function_with_content: ImageClassificationFunction, auth_test_user: OAuth2Renewer
) -> None:
    from_function: ImageClassificationFunction = image_classification_function_with_content
    to_function = NyckelFunctionDuplicator(from_function.function_id, auth_test_user, skip_confirmation=True)()
    assert set([s.external_id for s in to_function.list_samples()]) == set(
        [s.external_id for s in from_function.list_samples()]
    )
    assert set([s.name for s in to_function.list_labels()]) == set([s.name for s in from_function.list_labels()])

    to_function.delete()


def test_copy_text_function(
    text_classification_function_with_content: TextClassificationFunction, auth_test_user: OAuth2Renewer
) -> None:
    from_function: TextClassificationFunction = text_classification_function_with_content
    to_function = NyckelFunctionDuplicator(from_function.function_id, auth_test_user, skip_confirmation=True)()

    assert set([s.data for s in to_function.list_samples()]) == set([s.data for s in from_function.list_samples()])
    assert set([s.name for s in to_function.list_labels()]) == set([s.name for s in from_function.list_labels()])
    to_function.delete()


def test_copy_tabular_function(
    tabular_classification_function_with_content: TabularClassificationFunction, auth_test_user: OAuth2Renewer
) -> None:
    from_function: TabularClassificationFunction = tabular_classification_function_with_content
    to_function = NyckelFunctionDuplicator(from_function.function_id, auth_test_user, skip_confirmation=True)()

    assert set([json.dumps(s.data) for s in to_function.list_samples()]) == set(
        [json.dumps(s.data) for s in from_function.list_samples()]
    )
    assert set([s.name for s in to_function.list_labels()]) == set([s.name for s in from_function.list_labels()])
    to_function.delete()

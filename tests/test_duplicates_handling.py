from nyckel import TextClassificationFunction


def test_samples(text_classification_function: TextClassificationFunction) -> None:
    func = text_classification_function

    sample_ids = func.create_samples(["hello", "goodbye"])

    # Posting the same thing again, should return the same sample ids
    same_sample_ids = func.create_samples(["hello", "goodbye"])

    assert sample_ids == same_sample_ids

    # Now post a mix of exisitng and new samples

    new_sample_ids = func.create_samples(["hello", "goodday", "goodbye"])
    assert [new_sample_ids[0], new_sample_ids[2]] == sample_ids
    assert len(new_sample_ids) == 3

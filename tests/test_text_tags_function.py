import time

from nyckel import (
    ClassificationLabel,
    ClassificationPrediction,
    TagsAnnotation,
    TextTagsFunction,
    TextTagsSample,
)


def compare_tags_samples(sample1: TextTagsSample, sample2: TextTagsSample) -> bool:
    def remove_negative_annotations(sample: TextTagsSample) -> TextTagsSample:
        if sample.annotation:
            sample.annotation = [a for a in sample.annotation if a.present]
        return sample

    sample1.id = None
    sample2.id = None
    sample1 = remove_negative_annotations(sample1)
    sample2 = remove_negative_annotations(sample2)
    return sample1 == sample2


def test_labels(text_tags_function: TextTagsFunction) -> None:
    func = text_tags_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    label_ids = func.create_labels(labels_to_create)

    assert len(func.list_labels()) == 2

    func.delete_labels(label_ids[0:1])
    time.sleep(1)
    assert len(func.list_labels()) == 1


def test_unlabeled_samples(text_tags_function: TextTagsFunction) -> None:
    func = text_tags_function

    samples = ["This is a nice text", "This is a bad text"]

    func.create_samples(samples)
    time.sleep(1)
    assert len(func.list_samples()) == 2


def test_labeled_samples(text_tags_function: TextTagsFunction) -> None:
    func = text_tags_function

    samples = [
        TextTagsSample(data="This is a nice text", annotation=[TagsAnnotation(label_name="Nice", present=True)]),
        TextTagsSample(data="This is a bad text", annotation=[TagsAnnotation(label_name="Bad", present=True)]),
    ]

    sample_ids = func.create_samples(samples)
    time.sleep(1)
    samples_back = func.list_samples()
    assert len(samples_back) == 2
    first_sample = func.read_sample(sample_ids[0])
    assert compare_tags_samples(first_sample, samples[0])


def test_update_annotation(text_tags_function: TextTagsFunction) -> None:
    func = text_tags_function

    samples = [
        TextTagsSample(data="This is a nice text", annotation=[TagsAnnotation(label_name="Nice", present=True)]),
        TextTagsSample(data="This is a bad text", annotation=[TagsAnnotation(label_name="Bad", present=True)]),
    ]

    func.create_samples(samples)
    time.sleep(1)
    samples = func.list_samples()  # type: ignore
    samples[0].annotation = [
        TagsAnnotation(label_name="Nice", present=True),
        TagsAnnotation(label_name="Bad", present=True),
    ]
    func.update_annotation(samples[0])
    time.sleep(1)
    assert samples[0].id is not None
    updated_sample = func.read_sample(samples[0].id)
    assert compare_tags_samples(updated_sample, samples[0])


def test_delete_sample(text_tags_function: TextTagsFunction) -> None:
    func = text_tags_function

    samples = [
        TextTagsSample(data="This is a nice text", annotation=[TagsAnnotation(label_name="Nice", present=True)]),
        TextTagsSample(data="This is a bad text", annotation=[TagsAnnotation(label_name="Bad", present=True)]),
    ]

    sample_ids = func.create_samples(samples)
    time.sleep(1)
    func.delete_samples(sample_ids[0:1])
    time.sleep(1)
    assert len(func.list_samples()) == 1


def test_new_labels(text_tags_function: TextTagsFunction) -> None:

    func = text_tags_function

    func.create_labels([ClassificationLabel(name="label1")])
    time.sleep(0.5)
    sample_1_id = func.create_samples([TextTagsSample(data="sample1", annotation=[TagsAnnotation("label1")])])[0]

    func.create_labels([ClassificationLabel(name="label2")])
    time.sleep(0.5)
    sample_2_id = func.create_samples([TextTagsSample(data="sample2", annotation=[TagsAnnotation("label1")])])[0]

    sample1 = func.read_sample(sample_1_id)

    # When sample1 was created, label2 wasn't present yet.
    # So sample1 doesn't have an "opinion" about that label.
    # This means that it's not present in the annotation field.
    assert len(sample1.annotation) == 1
    assert TagsAnnotation("label1") in sample1.annotation
    assert TagsAnnotation("label2", present=False) not in sample1.annotation
    time.sleep(0.5)
    sample2 = func.read_sample(sample_2_id)
    # When sample2 was created, both labels were present.
    # Since we didn't give an annotation for label2, it was set to not present.
    assert len(sample2.annotation) == 2
    assert TagsAnnotation("label1") in sample2.annotation
    assert TagsAnnotation("label2", present=False) in sample2.annotation


def test_end_to_end(text_tags_function: TextTagsFunction) -> None:
    func = text_tags_function

    labels_to_create = [ClassificationLabel(name="Nice"), ClassificationLabel(name="Boo")]
    func.create_labels(labels_to_create)

    samples = [
        TextTagsSample(data="hello", annotation=[TagsAnnotation("Nice")]),
        TextTagsSample(data="hi", annotation=[TagsAnnotation(label_name="Nice")]),
        TextTagsSample(data="Good bye", annotation=[TagsAnnotation(label_name="Boo")]),
        TextTagsSample(data="I'm leaving", annotation=[TagsAnnotation(label_name="Boo")]),
        TextTagsSample(data="Hi again"),
    ]

    func.create_samples(samples)

    while not func.has_trained_model():
        print("-> No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    predictions = func.invoke(["Hej", "Hola", "Gruezi"])
    assert len(predictions) == 3
    for prediction in predictions:
        if len(prediction) > 0:
            assert isinstance(prediction[0], ClassificationPrediction)

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], TextTagsSample)

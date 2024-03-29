import time

from nyckel import ClassificationLabel, TextTagsFunction
from nyckel.functions.tags.tags import ClassificationPrediction, TagsAnnotation, TextTagsSample


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
    pred = func.invoke(["Howdy"])[0]
    assert isinstance(pred[0], ClassificationPrediction)

    assert len(func.invoke(["Hej", "Hola", "Gruezi"])) == 3

    returned_samples = func.list_samples()
    assert len(samples) == len(returned_samples)
    assert isinstance(returned_samples[0], TextTagsSample)

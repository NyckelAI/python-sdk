from nyckel import (
    TabularClassificationFunction,
    ClassificationLabel,
    TabularClassificationSample,
    ClassificationAnnotation,
    ClassificationPrediction,
)
import time
import pytest


def test_end_to_end(tabular_classification_function):
    func: TabularClassificationFunction = tabular_classification_function

    func.create_labels([ClassificationLabel(name="happy"), ClassificationLabel(name="sad")])

    func.create_samples(
        [
            TabularClassificationSample(
                data={"firstname": "Adam", "lastname": "Adame"}, annotation=ClassificationAnnotation(label_name="happy")
            ),
            TabularClassificationSample(
                data={"firstname": "Bo", "lastname": "Boden"}, annotation=ClassificationAnnotation(label_name="happy")
            ),
            TabularClassificationSample(
                data={"firstname": "Caren", "lastname": "Carrot"}, annotation=ClassificationAnnotation(label_name="sad")
            ),
            TabularClassificationSample(
                data={"firstname": "Dick", "lastname": "Denali"}, annotation=ClassificationAnnotation(label_name="sad")
            ),
        ]
    )

    fields = func._field_handler.list_fields()
    assert len(fields) == 2
    assert set([field.name for field in fields]) == set(["firstname", "lastname"])
    time.sleep(2)
    print(func.list_samples())
    assert len(func.list_samples()) == 4
    assert len(func.list_labels()) == 2

    while not func.has_trained_model():
        print("No trained model yet. Sleeping 1 sec...")
        time.sleep(1)

    prediction = func({"firstname": "Eric", "lastname": "Ebony"})
    assert isinstance(prediction, ClassificationPrediction)

    predictions = func.invoke(
        [{"firstname": "Eric", "lastname": "Ebony"}, {"firstname": "Frank", "lastname": "Froggy"}]
    )
    assert isinstance(predictions[0], ClassificationPrediction)


def test_field_creation(tabular_classification_function):
    func: TabularClassificationFunction = tabular_classification_function

    func.create_samples([TabularClassificationSample(data={"firstname": "Adam"})])
    fields = func._field_handler.list_fields()
    assert len(fields) == 1
    assert fields[0].type == "Text"
    func.create_samples([TabularClassificationSample(data={"age": 32})])
    fields = func._field_handler.list_fields()
    assert len(fields) == 2
    age_field = [field for field in fields if field.name == "age"][0]
    assert age_field.type == "Number"

    with pytest.raises(ValueError):
        # Can't assign a Text value to a Number field.
        func.create_samples([TabularClassificationSample(data={"age": "Twelve"})])

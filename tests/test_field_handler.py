import time

from nyckel.functions.classification.classification import TabularFunctionField
from nyckel.functions.classification.tabular_classification import TabularFieldHandler


def test_fields(tabular_classification_function, auth_test_user):
    func = tabular_classification_function
    # Just borrow the function_id and create a TabularFieldHandler directly to test it.
    fields_handler = TabularFieldHandler(func.function_id, auth_test_user)

    field = TabularFunctionField(name="Firstname", type="Text")
    field_id = fields_handler.create_fields([field])[0]
    fields_back = fields_handler.read_field(field_id)
    assert fields_back.name == "Firstname"
    assert fields_back.type == "Text"

    fields = fields_handler.list_fields()
    if len(fields) < 1:
        # Give the API a chance to catch up.
        time.sleep(1)
        fields = fields_handler.list_fields()
    assert len(fields) == 1

    # And delete label
    fields_handler.delete_field(field_id)
    fields = fields_handler.list_fields()
    if not len(fields) == 0:
        # Give the API a chance to catch up.
        time.sleep(1)
        fields = fields_handler.list_fields()
    assert len(fields) == 0

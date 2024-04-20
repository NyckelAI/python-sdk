import time

from nyckel import TabularFunctionField, TabularTagsFunction


class TestFields:

    def test_create(self, tabular_tags_function: TabularTagsFunction):
        func = tabular_tags_function
        func.create_fields(
            [
                TabularFunctionField(name="firstname", type="Text"),
                TabularFunctionField(name="height", type="Number"),
                TabularFunctionField(name="picture", type="Image"),
            ]
        )
        fields = func.list_fields()
        assert len(fields) == 3

    def test_read(self, tabular_tags_function: TabularTagsFunction):
        func = tabular_tags_function
        field_id = func.create_fields([TabularFunctionField(name="firstname", type="Text")])[0]
        time.sleep(0.5)

        field = func.read_field(field_id)
        assert field.name == "firstname"
        assert field.type == "Text"

    def test_delete(self, tabular_tags_function: TabularTagsFunction):
        func = tabular_tags_function
        field_id = func.create_fields([TabularFunctionField(name="firstname", type="Text")])[0]
        time.sleep(0.5)

        func.delete_field(field_id)
        time.sleep(0.5)

        fields = func.list_fields()
        assert len(fields) == 0

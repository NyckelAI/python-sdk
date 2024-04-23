# Multimodal classification

Nyckel supports multi-modal (joint) classification of images and text data through our Tabular function types. Use `TabularClassificationFunction` for binary and multi-class classification tasks, and `TabularTagsFunction` for multi-label tagging tasks.

## Binary and multi-class

Below is an e-commerce example:

``` py
from nyckel import Credentials, TabularClassificationFunction, TabularFunctionField, TabularClassificationSample, ClassificationAnnotation

credentials = Credentials(client_id="...", client_secret="...")

func = TabularClassificationFunction.create("Product categories", credentials)
func.create_fields([
    TabularFunctionField(name="Product", type="Text"),
    TabularFunctionField(name="Description", type="Text"),
    TabularFunctionField(name="Picture", type="Image")
])

# <image-data> entires can be provided as URLs, base64-encoded dataURIs, or local file paths.
func.create_samples(
    [
        TabularClassificationSample(
            data={"Product": "Acme wrench", "Description": "Solid steel; never breaks", "Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Hardware"),
        ),
        TabularClassificationSample(
            data={"Product": "ULINE Hammer", "Description": "Hammer with wood handle", "Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Hardware"),
        ),
        TabularClassificationSample(
            data={"Product": "Brooklinen bedding","Description": "Queen size bedding","Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Bedding"),
        ),
        TabularClassificationSamplefo(
            data={"Product": "Comforter", "Description": "Nice down comforter", "Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Bedding"),
        ),
    ]
)
```

## Multi-label (tagging)

Nyckel also supports multi-label tagging of images and text data through our Tabular Tags function type.

``` py
from nyckel import Credentials, TabularTagsFunction, TabularFunctionField, TabularTagsSample, TagsAnnotation

credentials = Credentials(client_id="...", client_secret="...")

func = TabularTagsFunction.create("Product categories", credentials)
func.create_fields([
    TabularFunctionField(name="Product", type="Text"),
    TabularFunctionField(name="Description", type="Text"),
    TabularFunctionField(name="Picture", type="Image")
])

# <image-data> entires can be provided as URLs, base64-encoded dataURIs, or local file paths.
func.create_samples(
    [
        TabularTagsSample(
            data={"Product": "Acme wrench", "Description": "Solid steel; never breaks", "Picture": "<image-data>"},
            annotation=[TagsAnnotation(label_name="Hardware"), TagsAnnotation(label_name="US-made")],
        ),
        TabularTagsSample(
            data={"Product": "ULINE Hammer", "Description": "Hammer with wood handle", "Picture": "<image-data>"},
            annotation=[TagsAnnotation(label_name="Hardware")],
        ),
        TabularTagsSample(
            data={"Product": "Brooklinen bedding","Description": "Queen size bedding","Picture": "<image-data>"},
            annotation=[TagsAnnotation(label_name="Bedding"), TagsAnnotation(label_name="Organic")],
        ),
        TabularTagsSample(
            data={"Product": "Comforter", "Description": "Nice down comforter", "Picture": "<image-data>"},
            annotation=[TagsAnnotation(label_name="Hardware")],
        ),
    ]
)
```

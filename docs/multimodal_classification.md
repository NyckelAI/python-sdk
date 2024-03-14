# Multimodal classification

Nyckel supports multi-modal (joint) classification of images and text data through our Tabular function type.

Below is an e-commerce example:

``` py
from nyckel import Credentials, TabularClassificationFunction, TabularFunctionField, TabularFunctionSample, ClassificationAnnotation

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
        TabularFunctionSample(
            data={"Product": "Acme wrench", "Description": "Solid steel; never breaks", "Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Hardware"),
        ),
        TabularFunctionSample(
            data={"Product": "ULINE Hammer", "Description": "Hammer with wood handle", "Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Hardware"),
        ),
        TabularFunctionSample(
            data={"Product": "Brooklinen bedding","Description": "Queen size bedding","Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Bedding"),
        ),
        TabularFunctionSample(
            data={"Product": "Comforter", "Description": "Nice down comforter", "Picture": "<image-data>"},
            annotation=ClassificationAnnotation(label_name="Bedding"),
        ),
    ]
)
```

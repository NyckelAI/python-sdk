# Nyckel Python SDK

[Nyckel](https://www.nyckel.com) is a Custom Classification API. It allows you to train, deploy and invoke custom [text](text_classification.md), [image](image_classification.md) and [tabular](tabular_classification.md) classification functions.

``` py
from nyckel import Credentials, TextClassificationFunction

cred = Credentials(client_id="...", client_secret="...")

func = TextClassificationFunction.create("IsToxic", cred)
func.create_samples([
    ("This is a nice comment", "Not toxic"),
    ("Hello friend", "Not toxic"),
    ("I hate people like you", "Toxic"),
    ("Who is this? Go away!", "Toxic"),
])

predictions = func.invoke(["This example is fantastic!"])
```

## Get started

* Visit [Nyckel](https://www.nyckel.com) and sign up for a free account
* Install the SDK: `pip install nyckel`
* Explore the SDK for [text](text_classification.md), [image](image_classification.md) and [tabular](tabular_classification.md) classification

# Nyckel

Python package for [www.nyckel.com](www.nyckel.com)

![build](https://github.com/NyckelAI/python-sdk/actions/workflows/build.yml/badge.svg)
![test](https://github.com/NyckelAI/python-sdk/actions/workflows/test.yml/badge.svg)
![docs](https://github.com/NyckelAI/python-sdk/actions/workflows/docs.yml/badge.svg)
[![PyPi version](https://img.shields.io/pypi/v/nyckel.svg)](https://pypi.python.org/pypi/nyckel/)

## Docs

Visit our [Docs pages](https://nyckelai.github.io/python-sdk/)

## Quickstart

```python
from nyckel import Credentials, TextClassificationFunction

# Get credentials from https://www.nyckel.com/console/keys
credentials = Credentials(client_id=..., client_secret=...)

# Create a new text classification function.
func = TextClassificationFunction.create(credentials=credentials, name="IsToxic")

# Provide a few examples.
func.create_samples([
    ("This is a nice comment", "not toxic"),
    ("Hello friend", "not toxic"),
    ("This is a bad comment", "toxic"),
    ("Who is this? Go away!", "toxic"),
])

# Nyckel trains and deploys your model in a few seconds...
# ...and you can start classifying text right away!

# Classify a new piece of text.
predictions = func.invoke(["New text"])
```

## Contributors

### Setup dev environment

```bash
pip install -r requirements.txt
```

### Building docs locally

Install packages

```bash
mkdocs==1.5.3
mkdocs-material==9.4.5
mkdocstrings==0.15.0
```

Run

```bash
mkdocs build
```

Note that the actual docs uses an [private repo](https://github.com/pawamoy-insiders/mkdocstrings-python) that allows cross-references for types.

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
from nyckel import OAuth2Renewer, TextClassificationFunction

# Get credentials from https://www.nyckel.com/console/keys
user = OAuth2Renewer(client_id=..., client_secret=...)

# Create a new text classification function.
func = TextClassificationFunction.new(user=user, name='IsToxic')

# Provide a few examples.
func.add_samples([
    ('This is a nice comment', 'not toxic'),
    ('Hello friend', 'not toxic'),
    ('This is a bad comment', 'toxic'),
    ('Who is this? Go away!', 'toxic'),
])

# Nyckel trains and deploys your model in a few seconds...
# ...and you can start classifying text right away!

# Classify a new piece of text.
label = func('New text')
```

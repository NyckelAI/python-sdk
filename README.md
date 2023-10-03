# Nyckel

Python package for <www.nyckel.com>

![build](https://github.com/NyckelAI/python-sdk/actions/workflows/build.yml/badge.svg)
![test](https://github.com/NyckelAI/python-sdk/actions/workflows/test.yml/badge.svg)
[![PyPi version](https://img.shields.io/pypi/v/nyckel.svg)](https://pypi.python.org/pypi/nyckel/)

## Usage

Retrieve `client_id` and `client_secret` from <https://www.nyckel.com/console/keys>

```python
from nyckel import OAuth2Renewer, TextClassificationFunction, TextClassificationSample

user = OAuth2Renewer(client_id=<CLIENT_ID>, client_secret=<CLIENT_SECRET>)

my_func = TextClassificationFunction.new('Example function name', user)

my_func.create_samples([TextClassificationSample(data="hello Nyckel")])
```

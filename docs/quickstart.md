# Quickstart

Use the Nyckel SDK to train, deploy and invoke a text classification function.

``` py
from nyckel import Credentials, TextClassificationFunction

# Get credentials from https://www.nyckel.com/console/keys
credentials = Credentials(client_id=..., client_secret=...)

# Create a new text classification function.
func = TextClassificationFunction.create(credentials=credentials, name='IsToxic')

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
predictions = func.invoke(["This example is fantastic!"])
```

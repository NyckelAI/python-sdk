# Copy Function

Use the Nyckel SDK to quickly copy your Nyckel function to a new function.

``` py
from nyckel import Credentials, TextClassificationFunction

# Get credentials from https://www.nyckel.com/console/keys
credentials = Credentials(client_id=..., client_secret=...)

# Load the source function.
from_func = TextClassificationFunction("<function_id>", credentials)

# Create a new, empty function.
to_func = TextClassificationFunction.create(f"{from_func.name} COPY", credentials)

# Copy samples over. Labels will automatically be created.
to_func.create_samples(from_func.list_samples())
```

# Delete the most recent samples

Uploaded a bunch of samples by mistake? No problem, you can delete them all at once.

``` py
from nyckel import Credentials, ImageClassificationFunction

# Load up  your function
credentials = Credentials(client_id="...", client_secret="...")
func = ImageClassificationFunction("<function_id>", credentials)

# Pull out the 500 more recent sample ids
samples = func.list_samples()
sample_ids = [sample.id for sample in samples][:500]

# Delete the samples
func.delete_samples(sample_ids)
```

# Delete label with samples

Deleting a label is easy in the [Nyckel UI](https://www.nyckel.com), [API](https://www.nyckel.com/docs#tags-delete-label) or indeed [this SDK](https://www.nyckel.com/docs/python-sdk). However, this leaves all samples in the function, but they are no longer associated with a label.

This guide shows how to *delete a single label along with all samples associated with it*.

``` py
from nyckel import Credentials, TextClassificationFunction

# This is the label name to be deleted
label_name = "" 

# Initialize your credentials and function
credentials = Credentials(client_id="...", client_secret="...")
func = TextClassificationFunction("<function_id>", credentials)

# Get all samples
samples = func.list_samples()

# Filter out the sample associated with the relevant label
samples = [sample for sample in samples if sample.annotation]
samples = [sample for sample in samples if sample.annotation.label_name == label_to_delete]

# Delete the samples
func.delete_samples([sample.id for sample in samples])

# Delete the label.
label_id_by_name = {label.name: label.id for label in func.list_labels()}
func.delete_labels([label_id_by_name[label_name]])
```

# Merge two labels

This guide shows how to merge two labels leaving all samples associated with the second label.

``` py
from nyckel import Credentials, TextClassificationFunction, ClassificationAnnotation, TextClassificationSample
import time

label_to_delete = ""  # This is the label name of the label to be deleted
label_to_keep = ""  # This is the label to which we want to assign samples from the first label

# Initialize your credentials and function
credentials = Credentials(client_id="...", client_secret="...")
func = TextClassificationFunction("<function_id>", credentials)

# Get all samples
samples = func.list_samples()

# Filter out the sample associated with the relevant label
samples = [sample for sample in samples if sample.annotation]
samples = [sample for sample in samples if sample.annotation.label_name == label_to_delete]

# Change the label of the relevant samples
for sample in samples:
    sample.annotation = ClassificationAnnotation(label_name=label_to_keep)
    func.update_annotation(sample)

# Pull samples again and assert that there are no samples associated with the first label.
time.sleep(2)  # Give the API a second to update all the way through.
samples = func.list_samples()
samples = [sample for sample in samples if sample.annotation]
samples = [sample for sample in samples if sample.annotation.label_name == label_to_delete]
assert len(samples) == 0, "Something went wrong. There are still samples associated with the first label."

# Delete the first label.
label_id_by_name = {label.name: label.id for label in func.list_labels()}
func.delete_labels([label_id_by_name[label_to_delete]])
```

# Pretrained classifiers

Nyckel offers a variety of [Pretrained Classifiers](https://www.nyckel.com/pretrained-classifiers/) for different use cases. These classifiers are trained and maintained by Nyckel, and ready for you to use.

## Setup

* Install the SDK: `pip install nyckel`
* Get your `client_id` and `client_secret` from [Nyckel](https://www.nyckel.com/console/keys)
* Find the pretrained classifier `function_id` you want by visiting our [Pretrained Classifiers Access Page](https://www.nyckel.com/console/pc-access)

## Classify text

``` py
import nyckel
function_id = 'toxic-language-identifier'
text = 'Hello friend!'
credentials = nyckel.Credentials(client_id, client_secret)
nyckel.invoke(function_id, text, credentials)
{
    'labelName': 'Not Offensive', 
    'labelId': 'label_3e46tk7jimpeozap', 
    'confidence': 0.9990668684599706
}
```

## Classify image

For image classification functions, use a `URL` or `base64 encoded image` as the input. Like so:

``` py
import nyckel
function_id = 'colors-identifier'
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Altja_j%C3%B5gi_Lahemaal.jpg/320px-Altja_j%C3%B5gi_Lahemaal.jpg"
credentials = nyckel.Credentials(client_id, client_secret)
nyckel.invoke(function_id, image_url, credentials)
{
    'labelName': 'Green',
    'labelId': 'label_csyoeshimzojn8xs',
    'confidence': 0.6614829725975897
}
```

## Reference

::: nyckel.invoke

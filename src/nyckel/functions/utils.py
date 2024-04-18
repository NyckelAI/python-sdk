from nyckel import ImageDecoder, ImageEncoder, ImageResizer, ImageSampleData
from nyckel.data_classes import NyckelId


def strip_nyckel_prefix(prefixed_id: str) -> str:
    split_id = prefixed_id.split("_")
    if len(split_id) == 2:
        return split_id[1]
    else:
        return prefixed_id


def is_nyckel_owned_url(url: str) -> bool:
    return url.startswith("https://s3.us-west-2.amazonaws.com/nyckel.server.")


class ImageSampleBodyTransformer:

    def __init__(self):
        self._decoder = ImageDecoder()
        self._encoder = ImageEncoder()
        self._resizer = ImageResizer()

    def __call__(self, sample_data: ImageSampleData) -> str:
        """Resizes if needed and encodes the sample data as a URL or dataURI."""
        if is_nyckel_owned_url(sample_data):
            # If the input points to a Nyckel S3 bucket, we know that the image is processed and verified.
            # In that case, we just point back to that URL.
            return sample_data

        if self._decoder.looks_like_url(sample_data):
            return self._encoder.to_base64(self._resizer(self._decoder.to_image(sample_data)))

        if self._decoder.looks_like_local_filepath(sample_data):
            return self._encoder.to_base64(self._resizer(self._decoder.to_image(sample_data)))

        if self._decoder.looks_like_data_uri(sample_data):
            return self._encoder.to_base64(self._resizer(self._decoder.to_image(sample_data)))

        raise ValueError(f"Can't parse input sample.data={sample_data}")


class ImageFieldTransformer:

    def __init__(self, field_id: NyckelId):
        self._field_id = field_id
        self._decoder = ImageDecoder()
        self._encoder = ImageEncoder()
        self._resizer = ImageResizer(max_image_size_pixels=384)

    def __call__(self, tabular_sample_data: dict) -> dict:
        if self._field_id in tabular_sample_data:
            tabular_sample_data[self._field_id] = self._sample_data_to_body(tabular_sample_data[self._field_id])
        return tabular_sample_data

    def _sample_data_to_body(self, sample_data: str) -> str:
        """Resizes if needed and encodes the sample data as a URL or dataURI."""
        if is_nyckel_owned_url(sample_data):
            # If the input points to a Nyckel S3 bucket, we know that the image is processed and verified.
            # In that case, we just point back to that URL.
            return sample_data

        if self._decoder.looks_like_url(sample_data):
            return self._encoder.to_base64(self._resizer(self._decoder.to_image(sample_data)))

        if self._decoder.looks_like_local_filepath(sample_data):
            return self._encoder.to_base64(self._resizer(self._decoder.to_image(sample_data)))

        if self._decoder.looks_like_data_uri(sample_data):
            return self._encoder.to_base64(self._resizer(self._decoder.to_image(sample_data)))

        raise ValueError(f"Can't parse input sample.data={sample_data}")

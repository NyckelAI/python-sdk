import base64
import os
from io import BytesIO
from typing import Tuple, Union

import pillow_avif  # type: ignore # noqa: F401. This is used transparently in PIL to support AVIF images.
import requests
from PIL import Image

from nyckel.config import MAX_IMAGE_SIZE_PIXELS


class ImageResizer:

    def __init__(self, max_image_size_pixels: int = MAX_IMAGE_SIZE_PIXELS):
        self._max_image_size_pixels = max_image_size_pixels

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self._needs_resize(img.width, img.height):
            return img
        new_width, new_height = self._get_new_width_height(img.width, img.height)
        img = img.resize((new_width, new_height))
        return img

    def _needs_resize(self, width: int, height: int) -> bool:
        return width > self._max_image_size_pixels or height > self._max_image_size_pixels

    def _get_new_width_height(self, width: int, height: int) -> Tuple[int, int]:
        if width > height:
            new_width = self._max_image_size_pixels
            new_height = int(new_width * height / width)
        else:
            new_height = self._max_image_size_pixels
            new_width = int(new_height * width / height)
        return new_width, new_height


class ImageDecoder:
    def to_image(self, sample_data: str) -> Image.Image:
        byte_stream = self.to_stream(sample_data)
        try:
            img = Image.open(byte_stream)
        except OSError:
            raise ValueError("Truncated Image Bytes")
        return img

    def to_stream(self, sample_data: str) -> BytesIO:
        if self.looks_like_url(sample_data):
            return self._load_from_url(sample_data)
        if self.looks_like_local_filepath(sample_data):
            return self._load_from_local_filepath(sample_data)
        if self.looks_like_data_uri(sample_data):
            return self._load_from_data_uri(sample_data)
        raise ValueError(f"Unable to parse input {sample_data=}.")

    def looks_like_url(self, sample_data: str) -> bool:
        return sample_data.startswith("https://") or sample_data.startswith("http://")

    def _load_from_url(self, url: str) -> BytesIO:
        response = requests.get(url, timeout=5)
        return BytesIO(response.content)

    def looks_like_local_filepath(self, local_path: str) -> bool:
        return os.path.exists(local_path)

    def _load_from_local_filepath(self, local_path: str) -> BytesIO:
        with open(local_path, "rb") as fh:
            return BytesIO(fh.read())

    def looks_like_data_uri(self, data_uri: str) -> bool:
        return data_uri.startswith("data:image")

    def _load_from_data_uri(self, data_uri: str) -> BytesIO:
        self._validate_image_data_uri(data_uri)
        image_b64_encoded_string = self.strip_base64_prefix(data_uri)
        im_bytes = base64.b64decode(image_b64_encoded_string)
        return BytesIO(im_bytes)

    def _validate_image_data_uri(self, inline_data: str) -> None:
        if inline_data == "":
            raise ValueError("Empty string")
        if "base64" not in inline_data:
            raise ValueError("base64 not in preamble.")
        inline_data_parts = inline_data.split(";base64,")
        if not len(inline_data_parts) == 2:
            raise ValueError("Unable to parse byte string.")
        if inline_data_parts[1] == "":
            raise ValueError("Empty image content")

    def strip_base64_prefix(self, inline_data: str) -> str:
        return inline_data.split(";base64,")[1]


class ImageEncoder:
    def to_base64(self, img: Union[Image.Image, BytesIO]) -> str:
        if isinstance(img, Image.Image):
            im_bytes = BytesIO()
            if img.mode == "P" and "transparency" in img.info:
                # Convert to RGBA if the image has transparency.
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                # Explicitly set RGBA backgrounds to white.
                background = Image.new("RGBA", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            if not img.mode == "RGB":
                img = img.convert("RGB")
            img.save(im_bytes, format="JPEG", quality=95)
        else:
            im_bytes = img
        encoded_string = base64.b64encode(im_bytes.getvalue()).decode("utf-8")
        return "data:image/jpg;base64," + encoded_string

from typing import Tuple

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

from typing import List


def chunkify_list(my_list: List, chunk_size: int) -> List[List]:
    return [
        my_list[start_index * chunk_size : (start_index + 1) * chunk_size]
        for start_index in range((len(my_list) + chunk_size - 1) // chunk_size)
    ]

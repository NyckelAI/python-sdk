def strip_nyckel_prefix(prefixed_id):
    split_id = prefixed_id.split("_")
    if len(split_id) == 2:
        return split_id[1]
    else:
        return prefixed_id

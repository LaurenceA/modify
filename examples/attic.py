
def flatten(d, prefix=()):
    """
    Analogue of dict.items() for nested dictionaries.
    Specifically, gives a flat list of keys and values.
    Values is just a list of tensors.
    Keys is more complex because it is a nested list.
    Specifically, keys is a list of one key for each value (tensor).
    Each key is a tuple of strings, representing a key for each nested dict.
    """
    keys = []
    values = []
    for k, v in d.items():
        full_key = prefix + (k,)
        if isinstance(v, dict):
            _keys, _values = flatten(v, prefix=full_key)
            keys = keys + _keys
            values = values + _values
        else:
            keys.append(full_key)
            values.append(v)
    return keys, values

def unflatten(keys, values):
    """
    Inverse of flatten
    """
    result = {}
    for key, value in zip(keys, values):
        #Reset to the top-level dict.
        pointer = result

        #Move the pointer to the last dict.
        #Create dicts as necessary.
        for k in key[:-1]:
            if k not in result:
                pointer[k] = {}
            pointer = pointer[k]

        pointer[key[-1]] = value
    return result

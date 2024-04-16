from collections import OrderedDict
from concurrent import futures
from functools import partial
from fuzzywuzzy import fuzz, process

def match_chackpoint(ckp1:OrderedDict, ckp2:OrderedDict, replacements:dict, num_worker=8, fast=False, full_output=False):
    keys1 = ckp1.keys()
    keys2 = ckp2.keys()
    keys_pool = set(keys2)
    results = OrderedDict()
    for key1 in keys1:
        shape1 = ckp1[key1].shape
        matched = []
        for key2 in keys_pool:
            shape2 = ckp2[key2].shape
            if shape2 == shape1:
                matched.append(key2)

        key_to_match:str = key1
        for old, new in replacements.items():
            if old in key_to_match:
                key_to_match = key_to_match.replace(old, new)
                
        match_func = partial(fuzz.token_sort_ratio, key_to_match)
        scores = list(map(match_func, matched))       
        result = sorted(zip(scores, matched), key=lambda x: x[0], reverse=True)

        if full_output:
            results[key1] = result
        else:
            results[key1] = result[0][1]

        if fast:
            keys_pool.remove(result[0][1])
    return results
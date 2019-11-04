import os
import sys
import numpy as np
import pandas as pd


def get_features_list(id_list):
    result = []
    for i in id_list:
        if ',' in i:
            result.extend(i.split(','))
        elif '.' in i:
            result.extend(i.split('.'))
        else:
            result.append(i)
    return result, list(set(result))


def get_occurrence(data, data_list):
    result = 0
    i = 0
    if '.' in data:
        data = data.split('.')
        for e in data:
            i += 1
            result += data_list.count(e)
        result = result/i
    elif ',' in data:
        data = data.split(',')
        for e in data:
            i += 1
            result += data_list.count(e)
        result = result/i
    else:
        result = data_list.count(data)

    return result




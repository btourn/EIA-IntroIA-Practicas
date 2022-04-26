dict1 = {'a': 1, 'b': 5, 'c': 10}
dict2 = {'a': 2, 'c': 1, 'd': 3}

dict3 = dict1.copy()
for key, val in dict2.items():
    if key in dict3:
        dict3[key] = dict2[key] + dict3[key]
    else:
        dict3[key] = val

print(dict3)
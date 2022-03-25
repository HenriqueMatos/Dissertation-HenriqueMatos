import inspect
import json

with open("config.json") as jsonFile:
    data = json.load(jsonFile)
final = {}
for item in data:
    # print(item)
    if type(data[item]) is dict:
        # print(list(data[item].keys()))
        final[item] = list(data[item].keys())
    else:
        final[item] = None
    # print(type(data[item]))
    # print(inspect.isclass(data[item]))
print(final)

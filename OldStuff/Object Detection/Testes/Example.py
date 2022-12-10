import numpy as np

def min_positive_integer_not_in_list(list):  # Our original array
  
    m = max(list)  # Storing maximum value
    if m < 1:
  
        # In case all values in our array are negative
        return 1
    if len(list) == 1:
  
        # If it contains only one element
        return 2 if list[0] == 1 else 1
    l = [0] * m
    for i in range(len(list)):
        if list[i] > 0:
            if l[list[i] - 1] != 1:
  
                # Changing the value status at the index of our list
                l[list[i] - 1] = 1
    for i in range(len(l)):
  
        # Encountering first 0, i.e, the element with least value
        if l[i] == 0:
            return i + 1
            # In case all values are filled between 1 and m
    return i + 2
  
# Driver Code
list = [0, 1,10, 2, -10, -20]
print(min_positive_integer_not_in_list(list))


out = np.identity(8, dtype=np.float32)
for index in range(4):
    out[index][4+index]=10.0
# out[0][4]=10
print(out)
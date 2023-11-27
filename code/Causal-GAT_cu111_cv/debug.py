import numpy as np
from evaluate import get_full_err_scores

array = np.random.rand(3, 4, 6)
i = 1
print(array)
if i == 0:
    print("case")

print(array)
test_labels = array[2, :, 0].tolist()
print(test_labels)

test_scores = get_full_err_scores(array)
print(test_scores)
last_column = test_scores[:, -1]
print(last_column)
indices = np.argmax(test_scores, axis=0).tolist()
print(indices)
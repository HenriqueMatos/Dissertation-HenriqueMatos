import tableprint as tp
import numpy as np
import time
import sys

sys.stdout.to_stdout("cona")

with tp.TableContext("ABC") as t:
    for _ in range(10):
        # time.sleep(0.1)
        t(np.random.randn(3,))
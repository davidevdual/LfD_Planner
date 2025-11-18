import numpy as np
import numpy.matlib
import os
import time
import Fast_Downward.fastdownward as fd

def compute_task_plan():
	start = time.time()
	task_planner = fd.FastDownward()
	end = time.time()
	print(end - start)
	return task_planner.get_plan()

if __name__ == "__main__":
	task_plan = compute_task_plan()
	print(task_plan)
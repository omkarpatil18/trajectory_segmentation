import cv2
import os
import glob
import pickle
import re
from task_constants import DATA_DIR

task_name = "sim_put_item_in_drawer"
dataset_path = os.path.join(DATA_DIR, task_name)
for file_name in glob.glob(os.path.join(dataset_path, "*.pickle")):
    with open(file_name, "rb") as file:
        print(file_name)
        demo = pickle.load(file)
        # first_img = cv2.cvtColor(demo['front_rgb'][0], cv2.COLOR_RGB2BGR)
        # last_img = cv2.cvtColor(demo['front_rgb'][-1], cv2.COLOR_RGB2BGR)
        # cv2.imshow('first_img', first_img)
        # cv2.imshow('last_img', last_img)
        # cv2.waitKey(1000)
        for i in range(len(demo["right_shoulder_rgb"])):
            front_rgb = cv2.cvtColor(demo["right_shoulder_rgb"][i], cv2.COLOR_RGB2BGR)
            cv2.imshow("right_shoulder_rgb", front_rgb)
            cv2.waitKey(5)

import os
import glob
import random
import pickle
from tqdm import tqdm
import sys

sys.path.append("/home/meribejayson/Desktop/Projects/realistic-imu/src")

from subjects.constants import NIMBLE_BODY_NODES_UIP, NIMBLE_BODY_NODES_DIP, NIMBLE_BODY_NODES_TOTAL_CAPTURE
from subjects.uip_subject import UIPSubject
from subjects.dip_imu_subject import DIPSubject
from subjects.total_capture_subject import TotalCaptureSubject

dip_subjects_dir = "/home/meribejayson/Desktop/Projects/realistic-imu/data/final_dataset/DIP"
uip_subjects_dir = "/home/meribejayson/Desktop/Projects/realistic-imu/data/final_dataset/UIP"
total_subjects_dir = "/home/meribejayson/Desktop/Projects/realistic-imu/data/final_dataset/Total-Capture"
GEOMETRY_PATH = "/home/meribejayson/Desktop/Projects/realistic-imu/data/final_dataset/Geometry/"



def get_b3d_files(directory):
    return glob.glob(os.path.join(directory, "**/*.b3d"), recursive=True)


def split_dataset(files, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2):
    random.shuffle(files)
    total = len(files)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    train = files[:train_size]
    dev = files[train_size:train_size + dev_size]
    test = files[train_size + dev_size:]

    return train, dev, test



dip_b3d_files = get_b3d_files(dip_subjects_dir)
uip_b3d_files = get_b3d_files(uip_subjects_dir)
#total_b3d_files = get_b3d_files(total_subjects_dir)

dip_train, dip_dev, dip_test = split_dataset(dip_b3d_files)
uip_train, uip_dev, uip_test = split_dataset(uip_b3d_files)
# total_train, total_dev, total_test = split_dataset(total_b3d_files)


"""
train_files = total_train + dip_train + uip_train
dev_files = total_dev + dip_dev + uip_dev
test_files = total_test + dip_test + uip_test
"""

train_files = dip_train + uip_train
dev_files = dip_dev + uip_dev
test_files = dip_test + uip_test

def generate_dataset(files, output_file):
    dataset = []
    for file in tqdm(files, desc="Processing mixed dataset", leave=True, disable=None):
        subject_class = None
        nimble_body_nodes = None

        if "DIP" in file:
            subject_class = DIPSubject
            nimble_body_nodes = NIMBLE_BODY_NODES_DIP
        elif "UIP" in file:
            subject_class = UIPSubject
            nimble_body_nodes = NIMBLE_BODY_NODES_UIP
        elif "Total-Capture" in file:
            subject_class = TotalCaptureSubject
            nimble_body_nodes = NIMBLE_BODY_NODES_TOTAL_CAPTURE

        subject = subject_class(file, GEOMETRY_PATH=GEOMETRY_PATH, nimble_body_nodes=nimble_body_nodes)
        dataset.extend(subject.get_subject_data())

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)


# Generate combined datasets
generate_dataset(train_files, "train.pkl")
generate_dataset(dev_files, "dev.pkl")
generate_dataset(test_files, "test.pkl")

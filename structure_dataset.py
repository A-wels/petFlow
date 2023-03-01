import argparse
from glob import glob
import os
import shutil
from tqdm import tqdm
from config import *
import random

def structure_dataset(path, target):
    print("Structure dataset from: ", path)
    print(path)
    list_of_files = os.listdir(path)
    # create list of all folders
    list_of_folders = []
    for folder in list_of_files:
        if folder.endswith("v") or folder.endswith("f"):
            list_of_folders.append(folder.split("_")[0] + "_" + folder.split("_")[1])
    list_of_folders = list(set(list_of_folders))
    print("Found ", len(list_of_folders), " Image series")
    random.shuffle(list_of_folders)

    for index, folder in tqdm(enumerate(list_of_folders), desc="Copying files...", total=len(list_of_folders)):
        stage = ""
        if index < len(list_of_folders) * PERCENTAGE_TRAIN:
            stage = "training"
        elif index < len(list_of_folders) * (PERCENTAGE_TRAIN + PERCENTAGE_VALID):
            stage = "validation"
        else:
           stage = "testing"
        
        target_path_clean = os.path.join(target,stage, "clean", folder)
        target_path_flow = os.path.join(target,stage, "flow", folder)

        if not os.path.exists(target_path_clean):
            os.makedirs(target_path_clean)
            os.makedirs(target_path_flow)
        # move file into target_path
        for file in glob(os.path.join(path, folder+"*")):
             if file.endswith("v"):
                target_path = os.path.join(target,stage, "clean", folder)
             else:
                target_path = os.path.join(target,stage, "flow", folder)

             shutil.copy(file, target_path)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', 
                        type=str,
                        dest='path',
                        nargs='?',
                        default=DEFAULT_DATASET_DIR,
                        help='Path of input data'
    )
    parser.add_argument('-t', 
                        type=str,
                        dest='target',
                        nargs='?',
                        default=DEFAULT_DATASET_TARGET,
                        help='Path target dir'
    )

    args = parser.parse_args()
    structure_dataset(args.path, args.target)

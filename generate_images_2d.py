import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.image
import shutil
from config import *


def load_data(path: str, extension: str, slice: str, dataset_name: str):
    # create directory for image data
    if not os.path.exists(DATASET_BASE_DIR):
     os.mkdir(DATASET_BASE_DIR)

    imagepath = os.path.join(DATASET_BASE_DIR,dataset_name)
    if os.path.exists(imagepath):
        print("Dataset already exists. Recreate? (Y/N)")
        answer = input()
        if answer == "y" or answer == "Y":
            shutil.rmtree(imagepath)
        else:
            exit()
    os.mkdir(imagepath)
    
        

    list_of_files = []
    datasets = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                list_of_files.append(f)
    
    list_of_files = sorted(list_of_files)
    
    for i, d in enumerate(list_of_files):
        if(d.split("_")[1] == slice):
            if i % 8 == 0:
                datasets.append([])
            datasets[-1].append(d)
    for dataset_id, dataset in enumerate(datasets):
        dataset_dir = os.path.join(imagepath, str(dataset_id))
        print("Saving dataset to: {}".format(dataset_dir))
        os.mkdir(dataset_dir)
        for frame_path in dataset:
            frame = np.reshape(np.fromfile(os.path.join(path,frame_path),'float32'),IMAGE_DIMENSION,order='F')
            p =  os.path.join(imagepath,str(dataset_id),"00000{}.png".format(frame_path.split("_")[2][:-2]))
            matplotlib.image.imsave(p, frame.T)
            exit()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', 
                        type=str,
                        dest='path',
                        nargs='?',
                        default=DATA_DIR,
                        help='Path of input data'
    )
    parser.add_argument('-e',
                        type=str,
                        dest='ext',
                        nargs='?',
                        default=DEFAULT_FILE_ENDING,
                        help='extenstion'
    )
    parser.add_argument('-s',
                        type=str,
                        dest='slice',
                        nargs='?',
                        default=DEFAULT_SLICE,
                        help='slice to use'
    )
    parser.add_argument('-n',
                        type=str,
                        dest='name',
                        nargs='?',
                        default=DEFAULT_DATASET_NAME,
                        help='dataset name'
    )
    args = parser.parse_args()

    load_data(args.path, args.ext, args.slice, args.name)

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from core.flowNetS import FlowNetS
import core.utils.flow_viz as flow_viz
import core.utils.frame_utils as frame_utils
import cv2
from visualize_gt_flow import create_gif, generate_vector_visualization
import os

from core.utils.frame_utils import read_gen

def load_model(model_path):
    model = FlowNetS()
    model = nn.DataParallel(model, [0])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# load the demo images from the given path
def load_demo_data(input_path):
    list_of_files = os.listdir(input_path)
    list_of_files.sort()
    file_names = []
    for file in list_of_files:
        p = os.path.join(input_path, file)
        if p.endswith(".v"):
            file_names.append(p)
    
    vector_fields = []
    vector_fields_unclamped = []
    for i in range(len(file_names)-1):
        img1 = read_gen(file_names[i])
        img2 = read_gen(file_names[i+1])

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()

        # set the 40 most left and right pixels to 0
        img1[0:80,:] = 0
        img1[-80:,:] = 0
        img2[0:80,:] = 0
        img2[-80:,:] = 0


        # clamp img1 and img 2 to [0,1]
        img1_c = torch.clamp(img1, 0,1)
        img2_c = torch.clamp(img2, 0,1)

        vector_fields_unclamped.append([img1, img2])
        vector_fields.append([img1_c, img2_c])
    # get filename: 000221_212_1.v -> 000221_212
    file_name = file_names[0].split('/')[-1].split('.')[0].split('_')[:-1]
    file_name = '_'.join(file_name)

    return vector_fields, file_name, vector_fields_unclamped



if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/optical_flow_2d.pt', help='path to the model')
    parser.add_argument('--input', type=str, default='demo_data', help='path to the input')
    parser.add_argument('--output', type=str, default='viz_results', help='path to the output')
    args = parser.parse_args()
   # load model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    # load model
    model = load_model(args.model_path).to(device)
    # set model for inference
    model.eval()

    # check if model is running on GPU
    if next(model.parameters()).is_cuda:
        print("Model is running on GPU")
    else:
        print("Model is running on CPU")

    print("Loaded model from {}".format(args.model_path))
    
    # load demo data
    vector_fields, file_name, vector_fields_unclamped = load_demo_data(args.input)

    # move demo data to GPU if available
    for i in range(len(vector_fields)):
        vector_fields[i][0] = vector_fields_unclamped[i][0].to(device)
        vector_fields[i][1] = vector_fields_unclamped[i][1].to(device)

    list_of_images = []
    # predict
    for i in tqdm(range(len(vector_fields))):
        img1 = vector_fields_unclamped[i][0]
        img2 = vector_fields_unclamped[i][1]

        input_images = torch.stack([img1,img2], dim=0)
        # add dimension for batch size
        input_images = input_images.unsqueeze(0)
        prediction = model(input_images)
        # remove added dimension for batch size
        prediction = prediction.squeeze(0).cpu().detach().numpy()/10
        prediction = np.transpose(prediction, (2,1,0))
        flow_dir = os.path.join(args.output,file_name,'flow')
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)
        frame_utils.writeFlow(os.path.join(flow_dir, '{}_{}.flo'.format(file_name, i+2)),prediction)
        flow_img = flow_viz.flow_to_image(prediction)
        image_path = os.path.join(args.output,file_name, 'image')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        
        import matplotlib
        background_path = os.path.join(image_path,'{}_{}_merge.png'.format(file_name, i+2))
        matplotlib.image.imsave(background_path, img1.cpu().T)

        output_path = os.path.join(image_path,'{}_{}.png'.format(file_name, i+2))

        generate_vector_visualization(prediction, flow_img, "{}_{}".format(file_name, i+2), output_path)

        from PIL import Image
        background = Image.open(background_path)

        background = background.convert("RGBA")
        flow_img_rgba = Image.fromarray(flow_img).convert("RGBA")

       # new_img = Image.blend(background, flow_img_rgba, 0.3)
       # new_img.save(background_path,"PNG")
       # cv2.imwrite(output_path, flow_img)
        list_of_images.append(output_path)

    create_gif(list_of_images,image_path, file_name)

    


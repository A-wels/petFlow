# Code adapted from: https://github.com/drinkingcoder/FlowFormer-Official

import numpy as np
from PIL import Image
from os.path import *
import re
import os
import cv2
from core.utils.flow_viz import flow_to_image
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)
def get_image_from_mvf(path):
        flow = read_gen(path)
        flow_img = flow_to_image(flow)     
        flow = np.transpose(flow, (1,0,2))
        flow_img = np.transpose(flow_img, (1,0,2))
        return flow, flow_img

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readFlow3d(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def convert_mvf_to_flo(mvf, mvf_name) -> str:
    mvf = np.reshape(mvf, [*[344,127],2], order='F')
    flo_name = mvf_name.replace('.mvf', '.flo')
    writeFlow(flo_name, mvf)
    return flo_name

def convert_3dmvf_to_flo(mvf):
    mvf = np.reshape(mvf, [*[344,344,127],3], order='F')
   # mvf = mvf[::2, ::2, :]
    return mvf

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    elif ext == '.v':
        v = np.fromfile(file_name, 'float32')
        is_3d = len(v) == 344*344*127
        if is_3d:
            # create image from generated data
            data =  np.reshape(v,[344,344,127], order='F')
            data = ((data - data.min()) * (1/(data.max() - data.min()) * 255)).astype('uint8')
            #data = data[::2, ::2, :]
        #  data = np.transpose(data, (1, 0))
            return data
        else:
            # create image from generated data
            data =  np.reshape(np.fromfile(file_name, 'float32'),[344,127], order='F')
            data = ((data - data.min()) * (1/(data.max() - data.min()) * 255)).astype('uint8')
        #  data = np.transpose(data, (1, 0))
            return data
    elif ext == '.mvf':
        mvf = np.fromfile(file_name, dtype=np.float32)
        is_3d = len(mvf) == 344*344*127*3
        if is_3d:
           return convert_3dmvf_to_flo(mvf)
        else:
            flo_name = file_name.replace('.mvf', '.flo')
            if os.path.exists(flo_name):
                flow = readFlow(flo_name).astype(np.float32)
                return flow
            # return np.transpose(flow, (1,0, 2))
            else:
                 return read_gen(convert_mvf_to_flo(mvf, file_name))
    return []

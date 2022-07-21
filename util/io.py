#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan

Tool  functions for file system operation and I/O. 
In the style of linux shell commands
'''
import os
import pickle as pkl
import subprocess
import logging
from . import strs, io
import math


def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created. 
    """
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
    return path

def make_parent_dir(path):
    """make the parent directories for a file."""
    parent_dir = get_dir(path)
    mkdir(parent_dir)
    
    
def pwd():
    return os.getcwd()

def dump(path, obj):
    path = get_absolute_path(path)
    parent_path = get_dir(path)
    mkdir(parent_path)
    with open(path, 'w') as f:
        logging.info('dumping file:' + path)
        pkl.dump(obj, f)

def load(path):
    path = get_absolute_path(path)
    with open(path, 'r') as f:
        data = pkl.load(f)
    return data

def join_path(a, *p):
    return os.path.join(a, *p)

def is_dir(path):
    path = get_absolute_path(path)
    return os.path.isdir(path)

is_directory = is_dir

def is_path(path):
    path = get_absolute_path(path)
    return os.path.ispath(path)
    
def get_dir(path):
    '''
    return the directory it belongs to.
    if path is a directory itself, itself will be return 
    '''
    path = get_absolute_path(path)
    if is_dir(path):
        return path;
    return os.path.split(path)[0]

def get_parent_dir(path):
    current_dir = get_dir(path)
    return get_absolute_path(join_path(current_dir, '..'))

def get_filename(path):
    return os.path.split(path)[1]

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def cd(p):
    p = get_absolute_path(p)
    os.chdir(p)
    
def ls(path = '.', suffix = None):
    """
    list files in a directory.
    return file names in a list
    """
    path = get_absolute_path(path)
    files = os.listdir(path)

    if suffix is None:       
        return files
        
    filtered = []
    for f in files:
        if string.ends_with(f, suffix, ignore_case = True):
            filtered.append(f)
    
    return filtered

def find_files(pattern):
    import glob
    return glob.glob(pattern)

def read_lines(p):
    """return the text in a file in lines as a list """
    p = get_absolute_path(p)
    f = open(p,'rU')
    return f.readlines()
    
def write_lines(p, lines, append_break = False):
    p = get_absolute_path(p)
    make_parent_dir(p)
    with open(p, 'w') as f:
        for line in lines:
            if append_break:
                f.write(line + '\n')
            else:
                f.write(line)

def cat(p):
    """return the text in a file as a whole"""
    cmd = 'cat ' + p
    return subprocess.getoutput(cmd)

def exists(path):
    path = get_absolute_path(path)
    return os.path.exists(path)

def not_exists(path):
    return not exists(path)

def load_mat(path):
    import scipy.io as sio
    path = get_absolute_path(path)
    return sio.loadmat(path)

def dump_mat(path, dict_obj, append = True):
    import scipy.io as sio
    path = get_absolute_path(path)
    make_parent_dir(path)
    sio.savemat(file_name = path, mdict =  dict_obj, appendmat = append)
    
def dir_mat(path):
    '''
    list the variables in mat file.
    return a list: [(name, shape, dtype), ...]
    '''
    import scipy.io as sio
    path = get_absolute_path(path)
    return sio.whosmat(path)
    
SIZE_UNIT_K = 1024
SIZE_UNIT_M = SIZE_UNIT_K ** 2
SIZE_UNIT_G = SIZE_UNIT_K ** 3
def get_file_size(path, unit = SIZE_UNIT_K):
    size = os.path.getsize(get_absolute_path(path))
    return size * 1.0 / unit
    
    
def create_h5(path):
    import h5py
    path = get_absolute_path(path)
    make_parent_dir(path)
    return h5py.File(path, 'w');

def open_h5(path, mode = 'r'):
    import h5py
    path = get_absolute_path(path)
    return h5py.File(path, mode);
    
def read_h5(h5, key):
    return h5[key][:]
def read_h5_attrs(h5, key, attrs):
    return h5[key].attrs[attrs]
    
def copy(src, dest):
    io.make_parent_dir(dest)
    import shutil
    shutil.copy(get_absolute_path(src), get_absolute_path(dest))
    
cp = copy

def remove(p):
    import os
    os.remove(get_absolute_path(p))
rm = remove

def search(pattern, path, file_only = True):
    """
    Search files whose name matches the give pattern. The search scope
    is the directory and sub-directories of 'path'. 
    """
    path = get_absolute_path(path)
    pattern_here = io.join_path(path, pattern)
    targets = []
    
    # find matchings in current directory
    candidates = find_files(pattern_here)
    for can in candidates:
        if io.is_dir(can) and file_only:
            continue
        else:
            targets.append(can)
            
    # find matching in sub-dirs
    files = ls(path)
    for f in files:
        fpath = io.join_path(path, f)
        if is_dir(fpath):
            targets_in_sub_dir = search(pattern, fpath, file_only)
            targets.extend(targets_in_sub_dir)
    return targets

def dump_json(path, data):
    import ujson as json
    path = get_absolute_path(path)
    make_parent_dir(path)

    with open(path, 'w') as f:
        json.dump(data, f)
    return path

def load_json(path):
    import ujson as json
    path = get_absolute_path(path)
    with open(path, 'r')  as f:
        return json.load(f)


def rotate(angle, x, y):
    """
    基于原点的弧度旋转

    :param angle:   弧度
    :param x:       x
    :param y:       y
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey

def xy_rorate(theta, x, y, centerx, centery):
    """
    针对中心点进行旋转

    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx + r_x, centery + r_y


def rec_rotate(x, y, width, height, theta):
    """
    传入矩形的x,y和宽度高度，弧度，转成QUAD格式
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = x + width / 2
    centery = y + height / 2

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x + width, y, centerx, centery)
    x3, y3 = xy_rorate(theta, x, y + height, centerx, centery)
    x4, y4 = xy_rorate(theta, x + width, y + height, centerx, centery)

    return x1, y1, x2, y2, x3, y3, x4, y4
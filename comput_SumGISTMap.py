import os
import pandas as pd
import numpy as np
import math
import time
import glob
import re
import pickle
import itertools

import time
from numba import jit, prange

def ReadPDB_ligand(path_pdb):
    amino_type, atom_type, atom_x, atom_y, atom_z = [],[],[],[],[]
    flag= False
    channel = 5+1
    with open(path_pdb,'r') as f:
        for line in f:
            if line[:3] == "TER":
                flag=True
                continue
            if not((line[:4] == "ATOM") or (line[:6] == "HETATM")):
                continue
            if line[17:20]=="WAT":
                continue

            amino = line[17:20]
            if flag:
                amino="LIGAND"
            amino_type.append(amino)
            atom_type.append(re.sub('\d+','',line[12:16].strip())[0])
            atom_x.append(float(line[31:38]))
            atom_y.append(float(line[39:46]))
            atom_z.append(float(line[47:54]))

    dict_ = {"H":"1", "C":"6", "N":"7", "O":"8", "F":"9", "P":"15", "S":"16","B":"35","I":"53"}
    for key, value in dict_.items():
        atom_type = [c.replace(key, value) for c in atom_type]
    

    return atom_x, atom_y, atom_z, atom_type, amino_type

def ReadPDB2(path_pdb):
    amino_type, atom_type, atom_x, atom_y, atom_z = [],[],[],[],[]
    atom_xyz = []
    flag= False
    channel = 5+1
    with open(path_pdb,'r') as f:
        for line in f:
            if line[:3] == "TER":
                flag=True
                continue
            if not((line[:4] == "ATOM") or (line[:6] == "HETATM")):
                continue
            if line[17:20]=="WAT":
                continue

            amino = line[17:20]
            if flag:
                amino="LIGAND"
            amino_type.append(amino)
            atom_type.append(re.sub('\d+','',line[12:16].strip())[0])
            xyz = []
            xyz.append(float(line[31:38]))
            xyz.append(float(line[39:46]))
            xyz.append(float(line[47:54]))
            atom_xyz.append(xyz)
            
    dict_ = {"H":"1", "C":"6", "N":"7", "O":"8", "F":"9", "P":"15", "S":"16","B":"35","I":"53"}
    for key, value in dict_.items():
        atom_type = [c.replace(key, value) for c in atom_type]
    atom_type = [int(c) for c in atom_type]   

    return atom_xyz, atom_type, amino_type

#原子ボクセル化　np配列使用
def voxelizer_atom_np(atoms_atomic_num, 
                atoms_xyz,
                atoms_amino,
                origin,
                lengths_index_voxelize,
                half_length_index_cutoff=5,
                length_voxel=0.5,
                factor=12,
                radiuses={6: 1.69984, 7: 1.62500, 8: 1.51369, 16: 1.78180, 1: 1.2},
                dtype=np.float64):

    atomic_num2index={index: i for i, index in enumerate(radiuses.keys())}
    diff_index_each_axis = range(-half_length_index_cutoff, half_length_index_cutoff + 1, 1)
    diff_index = np.array([row for row in itertools.product(diff_index_each_axis, 
                                                            diff_index_each_axis, 
                                                            diff_index_each_axis)]).astype(np.int64)

    #各軸の両端にcutoff値だけpadding領域を追加
    lengths_index_voxel_pad = lengths_index_voxelize + half_length_index_cutoff * 2
    voxel_pad = np.zeros((len(radiuses), *lengths_index_voxel_pad))  
    for i in range(len(atoms_atomic_num)):
        atom_atomic_num = atoms_atomic_num[i]
        atom_xyz = atoms_xyz[i]
        atom_amino = atoms_amino[i]
        if atom_amino == "LIGAND":
            continue
        atom_grid = displacement2index(atom_xyz - origin, length_voxel)
        atom_diff_grid_float = displacement2grid(atom_xyz - origin, length_voxel) % DELTA_INDEX
        
        distances = length_voxel * np.linalg.norm(
                                        diff_index - atom_diff_grid_float, 
                                        axis=1
                                        ).reshape(half_length_index_cutoff * 2 + 1, 
                                                half_length_index_cutoff * 2 + 1, 
                                                half_length_index_cutoff * 2 + 1)

        voxel_pad[
            atomic_num2index[atom_atomic_num],
            atom_grid[0]: atom_grid[0] + half_length_index_cutoff * 2 + 1,
            atom_grid[1]: atom_grid[1] + half_length_index_cutoff * 2 + 1,
            atom_grid[2]: atom_grid[2] + half_length_index_cutoff * 2 + 1
                ] += 1 - np.exp(-((radiuses[atom_atomic_num] / distances) ** factor))

    return voxel_pad[:, 
                    half_length_index_cutoff: -half_length_index_cutoff, 
                    half_length_index_cutoff: -half_length_index_cutoff, 
                    half_length_index_cutoff: -half_length_index_cutoff].astype(dtype)

DELTA_INDEX = 1


def displacement2index(displacement, length_voxel):
    return (displacement // length_voxel).astype(np.int64)

def displacement2grid(displacement, length_voxel):
    return displacement / length_voxel


def voxelizer_ligand_np(atoms_atomic_num, 
                atoms_xyz,
                atoms_amino,
                origin,
                lengths_index_voxelize,
                half_length_index_cutoff=5,
                length_voxel=0.5,
                factor=12,
                radiuses={6: 1.69984, 7: 1.62500, 8: 1.51369, 16: 1.78180, 1: 1.2},
                dtype=np.float64):

    atomic_num2index={index: 0 for _, index in enumerate(radiuses.keys())}
    diff_index_each_axis = range(-half_length_index_cutoff, half_length_index_cutoff + 1, 1)
    #itertools.product 使うと[001, 002, 003, ..., 998, 999]みたいに入る　
    diff_index = np.array([row for row in itertools.product(diff_index_each_axis, 
                                                            diff_index_each_axis, 
                                                            diff_index_each_axis)]).astype(np.int64)

    #各軸の両端にcutoff値だけpadding領域を追加？足りないのでは
    lengths_index_voxel_pad = lengths_index_voxelize + half_length_index_cutoff * 2
    #shape:(5,x_max-x_min+14*2, y_max-y_min+14*2, y_max-y_min+14*2)
    # *はリストのアンパック
    voxel_pad = np.zeros((1, *lengths_index_voxel_pad))  
    for i in range(len(atoms_atomic_num)):
        atom_atomic_num = atoms_atomic_num[i]
        atom_xyz = atoms_xyz[i]
        atom_amino = atoms_amino[i]
        if atom_amino != "LIGAND":
            continue
        atom_grid = displacement2index(atom_xyz - origin, length_voxel)
        #-DELTA_INDEX/2を無くすと河間・福島に合う。各ボクセルが示す座標はボクセルの中心に対し河間福島版はボクセルの原点側
        atom_diff_grid_float = displacement2grid(atom_xyz - origin, length_voxel) % DELTA_INDEX# - DELTA_INDEX / 2

        distances = length_voxel * np.linalg.norm(
                                        diff_index - atom_diff_grid_float, 
                                        axis=1
                                        ).reshape(half_length_index_cutoff * 2 + 1, 
                                                half_length_index_cutoff * 2 + 1, 
                                                half_length_index_cutoff * 2 + 1)


        # まず、0以外の距離での元の計算を実施
        nonzero_mask = (distances != 0)
        voxel_pad[0,
                  atom_grid[0]: atom_grid[0] + half_length_index_cutoff * 2 + 1,
                  atom_grid[1]: atom_grid[1] + half_length_index_cutoff * 2 + 1,
                  atom_grid[2]: atom_grid[2] + half_length_index_cutoff * 2 + 1
        ][nonzero_mask] += 1 - np.exp(-((radiuses[atom_atomic_num] / distances[nonzero_mask]) ** factor))

        # 次に、距離0の箇所は1に上書きする
        voxel_pad[0,
                  atom_grid[0]: atom_grid[0] + half_length_index_cutoff * 2 + 1,
                  atom_grid[1]: atom_grid[1] + half_length_index_cutoff * 2 + 1,
                  atom_grid[2]: atom_grid[2] + half_length_index_cutoff * 2 + 1
        ][~nonzero_mask] = 1
        
          
    return voxel_pad[:, 
                    half_length_index_cutoff: -half_length_index_cutoff, 
                    half_length_index_cutoff: -half_length_index_cutoff, 
                    half_length_index_cutoff: -half_length_index_cutoff].astype(dtype)

#len(file)とグリッド数、原点の座標を返す
def dxinfo(path_dx):
    with open(path_dx, 'r') as f:
        file = f.readlines()
    n_grid = file[0].strip().split()[5:8]
    origen = file[1].strip().split()[1:4]
    return len(file), np.array([int(n_grid[i])for i in range(3)]), np.array([float(origen[i])for i in range(3)])

def dx_to_ndarray(path_dx, len_file, n_grid):
    with open(path_dx, 'r') as f:
        file = f.readlines()
        
    #.dxファイルは7行目から値が書かれている。
    file = file[7:]
    
    #最後の一行にコメントがある場合除外する
    try:
        i = float(file[-1].replace("\n", "").split()[0])
    except ValueError as e:
        file = file[:-1]
        print(file[-1])
        i = float(file[-1].replace("\n", "").split()[0])
    
    table = [line.replace("\n", "").split() for line in file]
    table = [[float(s)for s in line] for line in table]
    df = pd.DataFrame(table)
    
    gist = df.values.ravel()
    gist = gist[~np.isnan(gist)]
    gist_vox= gist.reshape([int(n_grid[i])for i in range(3)], order='C')

    return gist_vox

def ReadDX(path_dx):
    len_file, grid, init = dxinfo(path_dx)
    voxel = dx_to_ndarray(path_dx, len_file, grid)
    return voxel

threshold = 1.0 / np.e

dxname="gist-dA"
protein = "4jwk"

path_dx_pred = "{}-{}-pred.dx".format(protein, dxname)
pred = ReadDX(path_dx_pred)
len_file, grid, init = dxinfo(path_dx_pred)

listfile = "/hogehoge/list.txt".format(protein)
with open (listfile) as f2:
    complex_names = f2.read().split()

for complex_name in complex_names:
    #Read the PDB file consisting of the protein and one water molecule.
    path_pdb = "/hogehoge/{}".format(complex_name)
        
    atom_xyz, atom_type, amino_type = ReadPDB2(path_pdb)

    channel = 5
    voxel = np.array([0.0]*grid[0]*grid[1]*grid[2]*channel).reshape(channel,grid[0], grid[1], grid[2])
    voxel_ligand = np.array([0.0]*grid[0]*grid[1]*grid[2]*1).reshape(1, grid[0], grid[1], grid[2])
    vdw = {1:1.20,6:1.69984,7:1.62500,8:1.51369,9:1.656,16:1.78180,15:1.871,35:1.978,53:2.094}
    cutoff = 10
    atoms = voxelizer_atom_np(atoms_atomic_num=atom_type, 
                              atoms_xyz=atom_xyz,
                              atoms_amino=amino_type,
                              origin=init,
                              lengths_index_voxelize=grid,
                              half_length_index_cutoff=cutoff,
                              length_voxel=0.5,
                              factor=12,   
    )
    ligand = voxelizer_ligand_np(atoms_atomic_num=atom_type, 
                                 atoms_xyz=atom_xyz,
                                 atoms_amino=amino_type,
                                 origin=init,
                                 lengths_index_voxelize=grid,
                                 half_length_index_cutoff=10,
                                 length_voxel=0.5,
                                 factor=12,   
                                 radiuses=vdw,
                                 dtype=np.float64)

    #Computation of dG_{W,replace}
    list_ = []
    gist = pred
    gist_vox_masked = np.where(ligand[0]>threshold, gist, 0)
    num_voxel_mask = np.where(ligand[0]>threshold, 1, 0).sum()
    value = gist_vox_masked.sum()
    list_.append(value)
 
    print(complex_name, list_[0])

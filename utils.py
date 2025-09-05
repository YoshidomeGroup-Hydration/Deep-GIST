import os
import numpy as np
import re
import itertools
import pandas as pd


class BaseBox:
    """
    タンパク質のボックス及びGIST-Mapのボックスに共通する基底のクラス。
    ボックスの持つ要素は
    self.box: ndarray, shape of (x,y,z,ch). 全体のボックス
    self.small_box: ndarray, shape of (n,x,y,z,ch). 分割後の部分ボックス
    self.box_recreate: ndarray, shape of (x,y,z,ch). 
        small_box からrecreate メソッドでボックスを再構成したボックスはここに格納される
        （注：予測結果のGIST-Mapを再構成する場合もこれに該当するため、予測結果はself.box ではなくself.box_recreate から取り出す）
    self.split_range: ndarray(int). 分割した際の分割数。recreate メソッドで再構成する際に用いる
    self.nvoxel_grid: ndarray(int). box 或いは box_recreate の大きさ
    self.init_coordinate: ndarray(float). ボックスの座標原点の座標
    self.dtype: numpy data type. ボックスを扱うデータタイプ（np.float16, np.float32, np.float64）。
    
    ボックスに要請される主な機能は 1. 読み込み 2. 分割 3. 再構成 であるため、それぞれをメソッドとして実装した。
    1. 読み込み
    Atoms と GISTMap クラスを参照
    2. 分割
    split, split_from_box
    3. 再構成
    recreate, recreate_from_small_box
    """
    def __init__(self, value="atoms"):
        self.order = "channellast" #channellast or channelfirst
        self.value = value #gist-dA, gist-dTStot, gist-Etot,...
        self.box = None
        self.small_box=None
        self.box_recreate=None
        self.split_range = np.array([0,0,0])
        self.nvoxel_grid = np.array([0,0,0])
        self.init_coordinate = np.array([0.,0.,0.])
        self.dtype = np.float16

    def load_box(self, box):
        self.box = box
        
    def recreate(self):
        """
        This function recreate the "box" from "small_box", which have the order of "channellast" (n,x,y,z,ch). 
        If self.order is "channelfirst", the order must be transposed.
        """

        X = self.split_range[0].astype(np.int64)
        Y = self.split_range[1].astype(np.int64)
        Z = self.split_range[2].astype(np.int64)
        
        if self.order == "channellast":
            small_box = self.small_box
        else:
            small_box = self.small_box.transpose(0,2,3,4,1)
        #print(small_box.shape)

        yz_list = []

        for x in range(X):
            z_list=[]
            for y in range(Y):
                voxel_z = np.concatenate([small_box[Z*Y*x + Z*y + z,16:32,16:32,16:32,:]for z in range(Z)],axis=2)#zが長いconcat->(16,16,160)
                z_list.append(voxel_z)
            voxel_yz = np.concatenate(z_list[:], axis=1)#yz平面が広いconcat->(16,160,160)
            #print(voxel_yz.shape)
            yz_list.append(voxel_yz)
        xyz = np.concatenate(yz_list[:], axis=0)
        #print(xyz.shape, data_range)
        
        
        box = np.array(xyz).astype(np.float32)
        if self.nvoxel_grid[0]>1:
            box = box[:self.nvoxel_grid[0], :self.nvoxel_grid[1], :self.nvoxel_grid[2]]
        if self.order == "channellast":
            self.box_recreate = box
        else:
            self.box_recreate = box[:,:,:,np.newaxis].transpose(3,0,1,2)
        
    
    def recreate_from_small_box(self, small_box, split_range=None, nvoxel_grid=None, ref = None, input_order="channellast"):
        #X = data_range[0].astype(np.int64)
        #Y = data_range[1].astype(np.int64)
        #Z = data_range[2].astype(np.int64)
        if input_order != "channellast":
            self.small_box = small_box.transpose(0,4,1,2,3)
        else:
            self.small_box = small_box
            
        if ref:
            self.split_range = ref.split_range
            self.nvoxel_grid = ref.nvoxel_grid
            self.init_coordinate = ref.init_coordinate
        else:
            try:
                self.split_range = np.array(split_range).astype(np.int64)
            except:
                print("split_range or reference BaseBox must be specified.")
            if nvoxel_grid:
                self.nvoxel_grid = nvoxel_grid
        self.recreate()

    
    
    #return (channel,index,48,48,48) data
    def split(self, filling="auto", training=False):
        """
        This function work with the order of "channelfirst".
        If self.order is "channellast", the order will be transposed.
        
        side_cut: If side_cut is True, the box is trimed before padding.
        """
        if training:
            self.trim()
        self.nvoxel_grid = self.box.shape[0:3]
        
        if filling=="auto":
            if self.value == "atoms":
                filling = 0.0
            elif self.value == "gist-dA":
                filling = -0.0398
            elif (self.value == "gist-gO") or (self.value=="gist-gH"):
                filling = 1.0
            else:
                filling = 0.0
                
        if self.order=="channellast":
            data = self.box.transpose(3,0,1,2)
        else:
            data = self.box
        
        data_pad = self.pad(data, filling=filling)
        point_x = [16*i for i in range(data_pad.shape[1]//16 - 2)]
        point_y = [16*i for i in range(data_pad.shape[2]//16 - 2)]
        point_z = [16*i for i in range(data_pad.shape[3]//16 - 2)]
        #print('point',point_x, point_y, point_z)

        voxel_list = []
        for channel in range(data_pad.shape[0]):
            voxel_monochannel = []
            for px in point_x:
                for py in point_y:
                    for pz in point_z:

                        voxel = []
                        voxel = data_pad[channel,px:px+48, py:py+48, pz:pz+48]
                        voxel_monochannel.append(voxel)
            voxel_list.append(voxel_monochannel)
        range_list = [len(point_x), len(point_y), len(point_z)]
        #print('split into {} voxels'.format(len(voxel_list[0])), '({}*{}*{} {}ch)'.format(range_list[0], range_list[1], range_list[2], len(voxel_list)))
        #print(range_list)
        self.split_range = np.array(range_list)
        if self.order=="channellast":
            self.small_box = np.array(voxel_list).transpose(1,2,3,4,0).astype(self.dtype)
        else:
            self.small_box = np.array(voxel_list).transpose(1,0,2,3,4).astype(self.dtype)
    
    def split_from_box(self, box, filling="auto", side_cut=True, input_order="channel_last"):
        if input_order == "channellast":
            self.box = box
        else:
            self.box = box.transpose(1,2,3,0)
        self.split(filling=filling, side_cut=side_cut)
        
    def pad(self, data, filling):
        """
        This function work with the order of "channelfirst".
        side_cut: If side_cut is True, the box is trimed before padding.
        """
        
        #if data.shape[0]==1: filling = 1.0
        #else: filling = 0.0

        nchannel = data.shape[0]
        padding_range = [32-data.shape[1]%16, 32-data.shape[2]%16, 32-data.shape[3]%16]
        #padding_range = [143-water.shape[0], 143-water.shape[1], 143-water.shape[2]]

        padding_x_start = np.full((nchannel,16, data.shape[2], data.shape[3]), filling)
        padding_x_end = np.full((nchannel,padding_range[0], data.shape[2], data.shape[3]), filling)
        data_pad = np.concatenate([padding_x_start, data, padding_x_end], axis=1)

        padding_y_start = np.full((nchannel,data_pad.shape[1], 16, data_pad.shape[3]), filling)
        padding_y_end = np.full((nchannel,data_pad.shape[1], padding_range[1], data_pad.shape[3]), filling)
        data_pad = np.concatenate([padding_y_start, data_pad, padding_y_end], axis=2)

        padding_z_start = np.full((nchannel,data_pad.shape[1], data_pad.shape[2], 16), filling)
        padding_z_end = np.full((nchannel,data_pad.shape[1], data_pad.shape[2], padding_range[2]), filling)
        data_pad = np.concatenate([padding_z_start, data_pad,padding_z_end], axis=3)

        #print('padding_range:', padding_range)
        #print('adjusted to ',data_pad.shape, end='   ')
        return data_pad

    def trim(self):
        self.box = self.box[3:-3, 3:-3, 3:-3,:]

    def dxinfo(self):
        with open(self.path_dx, 'r') as f:
            file = f.readlines()
        n_grid = file[0].strip().split()[5:8]
        origen = file[1].strip().split()[1:4]
        #print('grid: ', n_grid, ' origen: ', origen)
        self.len_file = len(file)
        self.nvoxel_grid = np.array([int(n_grid[i])for i in range(3)])
        self.init_coordinate = np.array([float(origen[i])for i in range(3)])


class Atoms(BaseBox):
    def __init__(self, init_coordinate=(0.,0.,0.), length_cutoff=8., length_voxel=0.5):
#    def __init__(self, init_coordinate=(0.,0.,0.), length_cutoff=5., length_voxel=0.5):
        """
        init_coordinate(float): DXファイルの基準座標
        length_cutoff(float): 原子の影響範囲のカットオフ[Å]。原子の影響が計算される nvoxel は length_cutoff/length_voxel
        """
        
        super().__init__(value="atoms")
        DELTA_INDEX = 1
        self.length_cutoff = length_cutoff #Added by TY
        self.length_voxel = length_voxel
        self.nvoxel_cutoff = int(length_cutoff/self.length_voxel)
        order_atype = "fukushima"
        if order_atype == "kawama":
            self.radiuses = {6: 1.69984, 7: 1.62500, 8: 1.51369, 16: 1.78180, 1: 1.2}
        else:
            self.radiuses = {1: 1.2, 6: 1.69984, 7: 1.62500, 8: 1.51369, 16: 1.78180}
            

    def voxelize(self,
                 protein,
                 path_pdb,
                 path_dx=None,
                 factor=12,
                 dtype=np.float64):
        self.protein = protein
        self.path_pdb = path_pdb
        self.atoms_xyz, self.atom_type, self.amino_type = self.read_pdb(self.protein, self.path_pdb)
        self.path_dx = path_dx
        
        if path_dx:
            self.dxinfo()
        else:
            #Revised by TY
            mins = self.atoms_xyz.min(axis=0)  # [xmin, ymin, zmin]            
            maxs = self.atoms_xyz.max(axis=0)  # [xmax, ymax, zmax]                      
            xmin, ymin, zmin = mins[0], mins[1], mins[2]
            xmax, ymax, zmax = maxs[0], maxs[1], maxs[2]
            
            self.nvoxel_grid = np.array([int(xmax-xmin)*2+self.nvoxel_cutoff*2, int(ymax-ymin)*2+self.nvoxel_cutoff*2,int(zmax-zmin)*2+self.nvoxel_cutoff*2])
            self.init_coordinate = np.array([xmin - self.length_cutoff, ymin - self.length_cutoff, zmin - self.length_cutoff])
#            self.init_coordinate = np.array([0.,0.,0.])
        
            
        atomic_num2index={index: i for i, index in enumerate(self.radiuses.keys())}
        diff_index_each_axis = range(-self.nvoxel_cutoff, self.nvoxel_cutoff + 1, 1)
        #itertools.product 使うと[001, 002, 003, ..., 998, 999]みたいに入る　
        diff_index = np.array([row for row in itertools.product(diff_index_each_axis, 
                                                                diff_index_each_axis, 
                                                                diff_index_each_axis)]).astype(np.int64)

        #各軸の両端にcutoff値だけpadding領域を追加
        nvoxel_grid_pad = self.nvoxel_grid + self.nvoxel_cutoff * 4
        #shape:(5,x_max-x_min+14*2, y_max-y_min+14*2, y_max-y_min+14*2)
        # *はリストのアンパック
        box_pad = np.zeros((len(self.radiuses), *nvoxel_grid_pad))  
        #print(box_pad.shape, self.nvoxel_grid)
        for i in range(len(self.atom_type)):
            atom_atomic_num = self.atom_type[i]
            atom_xyz = self.atoms_xyz[i]
            atom_amino = self.amino_type[i]
            if atom_amino == "LIGAND":
                continue
            atom_grid = self.displacement2index(atom_xyz - self.init_coordinate, self.length_voxel) + 2*self.nvoxel_cutoff
            atom_diff_grid_float = self.displacement2grid(atom_xyz - self.init_coordinate, self.length_voxel) % 1

            distances = self.length_voxel * np.linalg.norm(
                                            diff_index - atom_diff_grid_float, 
                                            axis=1
                                            ).reshape(self.nvoxel_cutoff * 2 + 1, 
                                                    self.nvoxel_cutoff * 2 + 1, 
                                                    self.nvoxel_cutoff * 2 + 1)
            #print(atom_grid, atom_grid[0] + self.nvoxel_cutoff, atom_grid[1] + self.nvoxel_cutoff, atom_grid[2] + self.nvoxel_cutoff)

            box_pad[
                atomic_num2index[atom_atomic_num],
                atom_grid[0]-self.nvoxel_cutoff: atom_grid[0] + self.nvoxel_cutoff + 1,
                atom_grid[1]-self.nvoxel_cutoff: atom_grid[1] + self.nvoxel_cutoff + 1,
                atom_grid[2]-self.nvoxel_cutoff: atom_grid[2] + self.nvoxel_cutoff + 1
                    ] += 1 - np.exp(-((self.radiuses[atom_atomic_num] / distances) ** factor))
            

        self.box = box_pad[:, 
                                         2*self.nvoxel_cutoff: -2*self.nvoxel_cutoff, 
                                         2*self.nvoxel_cutoff: -2*self.nvoxel_cutoff, 
                                         2*self.nvoxel_cutoff: -2*self.nvoxel_cutoff].astype(dtype)
        if self.order=="channellast":
            self.box = self.box.transpose(1,2,3,0)
        
        """return voxel_pad[:, 
                        self.nvoxel_cutoff: -self.nvoxel_cutoff, 
                        self.nvoxel_cutoff: -self.nvoxel_cutoff, 
                        self.nvoxel_cutoff: -self.nvoxel_cutoff].astype(dtype)
        """


    def read_pdb(self, protein, path_pdb):
        """
        PDBを読み込むメソッド
        Input
        path_pdb: .pdb のパス

        Output
        atom_x_: 各原子のx座標
        atom_type: 各原子の原子種名（H,C,N,O,S）
        amino_type: 各原子のアミノ酸種名、リガンドの場合は"LIGAND"
        """
        self.protein = protein
        self.path_pdb = path_pdb
        amino_type, atom_type = [],[]
        atoms_xyz = []
        flag= False
        with open(path_pdb,'r') as f:
            for line in f:
                info = re.split(" +",line)
                head = info[0]
                
                if not((head == "ATOM") or (head == "HETATM")):
                    continue
                amino = info[3]
                if amino == "WAT":
                    continue
                if head == "HETATM":
                    amino="LIGAND"
                    
                amino_type.append(amino)
                atom_type.append(re.sub('\d+','',info[2])[0])
                xyz = []
                xyz.append(float(info[5]))
                xyz.append(float(info[6]))
                xyz.append(float(info[7]))
                atoms_xyz.append(xyz)
        atoms_xyz = np.array(atoms_xyz)
        dict_ = {"H":"1", "C":"6", "N":"7", "O":"8", "F":"9", "P":"15", "S":"16","B":"35","I":"53"}
        for key, value in dict_.items():
            atom_type = [c.replace(key, value) for c in atom_type]
        atom_type = [int(c) for c in atom_type]
        return atoms_xyz, atom_type, amino_type



    def displacement2index(self,displacement, length_voxel):
        return (displacement // length_voxel).astype(np.int64)

    def displacement2grid(self,displacement, length_voxel):
        return displacement / length_voxel
 
class GistMap(BaseBox):
    def __init__(self, value="gist-dA"):
        super().__init__(value=value)
        
    def read_dx(self, path_dx):
        self.path_dx = path_dx
        self.dxinfo()
        box = self.dx_to_ndarray()
        self.box = box[:,:,:,np.newaxis]


    def dx_to_ndarray(self): 
        len_file, n_grid = self.len_file, self.nvoxel_grid
        with open(self.path_dx, 'r') as f:
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

        #旧式
        """
        skip = [i for i in range(7)]
        skip.append(len_file)
        skip.append(len_file -1)
        #dx_file_name = 'gist-gO.dx'
        df = pd.read_csv(path_dx,
                         header=None, sep = ' ',
                         skiprows=skip,
                         usecols=[0,1,2]
                        )

        gist = df.values.ravel()
        gist = gist[~np.isnan(gist)]
        gist_vox= gist.reshape([int(n_grid[i])for i in range(3)], order='C')
        """
        return gist_vox

    def to_dx(self, box=None, save_dir=None):
        if not box:
            box = self.box_recreate
        if not save_dir:
            save_dir="./{}.dx".format(self.value)
        init = self.init_coordinate
        nx, ny, nz = box.shape[0], box.shape[1], box.shape[2]

        with open(save_dir, "w") as f:
            f.write("object 1 class gridpositions counts {} {} {}\n".format(nx, ny, nz))
            f.write("origin {:15.8f}{:15.8f}{:15.8f}\n".format(init[0], init[1], init[2]))
            f.write("delta       0.50000000 0 0\ndelta  0      0.50000000 0\ndelta  0 0      0.50000000\n")
            f.write("object 2 class gridconnections counts{:9d}{:9d}{:9d}\n".format(nx, ny, nz))
            f.write("object 3 class array type double rank 0 items {:27d} data follows\n".format(nx * ny * nz))
            idx=0
            amari = (nx * ny * nz) % 3
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        idx+=1
                        moji = "0." + "{:.4E}".format(box[x, y, z, 0]*10.).replace(".","").replace("E+","E+0").replace("E-","E-0")
                        try : 
                            if moji.find("0.-") != -1 : moji = re.sub("0.-(....).E", "-0.\\1E", moji)
                            if moji[moji.rfind("E")+5] != None : moji = moji.replace("+0","+").replace("-0","-")
                        except : pass
                        if idx % 3 == 0 :
                            f.write(moji+"\n")
                        else : 
                            f.write(moji + " ")
                        if nx * ny * nz == idx and amari!=0:
                            f.write("\n")

def trim(box, edge=3):
    return box[edge:-edge, edge:-edge, edge:-edge]

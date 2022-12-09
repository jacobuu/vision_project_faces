import pyvista as pv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import norm
import sys
import os

def landmks_load(filename):
    with open(filename, 'r') as f:
        Lines = f.readlines()

    count = 0
    ldks = []
    ldks_ID = []
    # parse lines
    if Lines[0].find('l') == -1:
        for line in Lines:
                count += 1 
                x, y, z = line.split(' ')
                ldks_ID.append(np.int32(count))
                ldks.append([np.float32(x), np.float32(y), np.float32(z)])
    return np.asarray(ldks), np.asarray(ldks_ID)

def mesh_load(obj_file):
    reader = pv.get_reader(obj_file)
    mesh = reader.read()
    assert mesh.is_all_triangles()
    mesh["ID"] = [i for i in range(mesh.n_points)] # add IDs
    return mesh

def mesh_plot(mesh, tex_file):
    tex = pv.read_texture(tex_file)
    
    p = pv.Plotter(notebook=True)
    p.add_mesh(mesh, texture=tex, show_edges=True, pickable=True)
    
    p.show(cpos='xy')

def landmarks_plot(landmks):
    poly = pv.PolyData()
    poly.points = landmks
    cells = np.full((len(landmks) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(landmks) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(landmks), dtype=np.int_)
    poly.lines = cells
    return poly

def compute_metrics(landmks, landmks_TRUE):
    print(f"Computing metrics...")
    sum = 0
    for i in range(0,50):
        error=0
        for coord in range(0,3):
            error += (landmks[i][coord]-landmks_TRUE[i][coord])**2
        sum += np.sqrt(error)
    mean = sum/50
    return mean


def main(argv):
    if len(argv) != 2:
        print("Insert the path of the directory containing the face to be rendered")
        return
    
    path = str(argv[1])
    if not os.path.isdir(path):
        print("You MUST specify the path of a directory")
        return
    _, dir_name = os.path.split(path)
    if not os.path.exists(path + '/' + dir_name + '.obj'):
        print(f"In the specified directory there is not the {dir_name}.obj file")
        return
    if not os.path.exists(path + '/' + dir_name + '_landmarks.txt'):
        print(f"You must use: python predict.py --c configs\CUSTOM-RGB+depth.json --n {path}/{dir_name}.obj")
        return
    # if not os.path.exists(path + '/' + dir_name + '_landmarks.vtk'):
    #     print(f"You must use: python predict.py --c configs\CUSTOM-RGB+depth.json --n {path}/{dir_name}.obj")
    #     return

    DATA_DIR = path
    face = dir_name

    pv.set_jupyter_backend('pythreejs')
    l1_obj = f'{DATA_DIR}/{face}.obj'
    l1_img = f'{DATA_DIR}/{face}.jpg'
    file_lks = f'{DATA_DIR}/{face}_landmarks.txt'
    file_true_lks = f'{DATA_DIR}/{face}.txt'
    # load mesh from obj file
    mesh = mesh_load(l1_obj)

    # load landmarks
    landmks, ldks_ID = landmks_load(file_lks)
    landmks_TRUE, ldks_TRUE_ID = landmks_load(file_true_lks)
    RMSE = compute_metrics(landmks, landmks_TRUE)
    print(f"RMSE: {RMSE}")
    poly2 = pv.PolyData(landmks_TRUE)
    poly2["id"] = ldks_TRUE_ID
    poly = pv.PolyData(landmks)
    poly["id"] = ldks_ID

    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=False, color='white')
    pl.add_point_labels(poly, "id", point_size=1.0, text_color='white', shape_color='red')
    pl.add_point_labels(poly2, "id", point_size=1.0, text_color='white', shape_color='black')
    pl.camera_position = 'xy'
    pl.show()

if "__main__":
    main(sys.argv)
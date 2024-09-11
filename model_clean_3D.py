"""
Version: 1.5

Summary: Statistical outlier removal for 3d point cloud

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 model_clean_3D.py -i ~/example/test.ply -o ~/example/result/ --outlier_ratio


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")


output:
*.xyz: xyz format file only has 3D coordinates of points 
*_aligned.ply: aligned model with only 3D coordinates of points 

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np 
import argparse

import os
import sys
import open3d as o3d
import copy
import pathlib
from scipy.spatial.transform import Rotation as Rot
import math



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


#compute angle between two vectors(works for n-dimensional vector),
def dot_product_angle(v1,v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        
        print("Zero magnitude vector!")
        
        return 0
        
    else:
        vector_dot_product = np.dot(v1,v2)
        
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        if np.degrees(arccos) > 90:
            
            angle = np.degrees(arccos) - 90
        else:
            
            angle = np.degrees(arccos)
    
    return angle
        


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions


# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')

    return file_path, filename, basename





def format_converter(model_file):
    

    # Pass xyz to Open3D.o3d.geometry.PointCloud 

    pcd = o3d.io.read_point_cloud(model_file)
    
    
    #filename = current_path + base_name + '_binary.ply'
    
    #o3d.io.write_point_cloud(filename, pcd)
    
    #pcd = o3d.io.read_point_cloud(filename)
    
    #print(np.asarray(pcd.points))
    
    #visualize the original point cloud
    #o3d.visualization.draw_geometries([pcd])
    
    Data_array = np.asarray(pcd.points)
    
    color_array = np.asarray(pcd.colors)
    
    #print(len(color_array))
    
    
    
    #color_array[:,2] = 0.24
    
    #pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    #o3d.visualization.draw_geometries([pcd])
    
    #pcd.points = o3d.utility.Vector3dVector(points)

    # threshold data
    
    if len(color_array) == 0:
        
        pcd_sel = pcd
    else:
        pcd_sel = pcd.select_by_index(np.where(color_array[:, 2] > outlier_ratio)[0])
    
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd_sel])


    # copy original point cloud for rotation
    pcd_cleaned = copy.deepcopy(pcd_sel)
    
    
    # get the model center postion
    model_center = pcd_cleaned.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_cleaned.translate(-1*(model_center))
    
    
    # Statistical outlier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    

    # visualize the oulier removal point cloud
    print("Statistical outlier removal\n")
    cl, ind = pcd_cleaned.remove_statistical_outlier(nb_neighbors = 100, std_ratio = 0.001)
    #display_inlier_outlier(pcd_r, ind)
    
    

    
    #print("Statistical outlier removal\n")
    #cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 40, std_ratio = 0.00001)
    #display_inlier_outlier(pcd_r, ind)
   
    

    return pcd_cleaned








if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest="input", type=str, required=True, help="full path to 3D model file")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ap.add_argument("--outlier_ratio", required = False, type = float, default = 0.1, help = "outlier remove ratio")
    args = vars(ap.parse_args())



     # single input file processing
    ###############################################################################
    if os.path.isfile(args["input"]):

        input_file = args["input"]

        (file_path, filename, basename) = get_file_info(input_file)

        print("Compute {} model orientation and aligning models...\n".format(file_path, filename, basename))

        # result path
        result_path = args["output_path"] if args["output_path"] is not None else file_path

        result_path = os.path.join(result_path, '')

        # print out result path
        print("results_folder: {}\n".format(result_path))
        
        # parameter
        outlier_ratio = args["outlier_ratio"]


        # start pipeline
        ########################################################################################
        # model alignment 
        pcd_cleaned = format_converter(input_file)
        
        
        
        ####################################################################
        # write aligned 3d model as ply file format
        # get file information

        #Save model file as ascii format in ply
        result_filename = result_path + basename + '_cleaned.ply'

        #write out point cloud file
        o3d.io.write_point_cloud(result_filename, pcd_cleaned, write_ascii = True)

        # check saved file
        if os.path.exists(result_filename):
            print("Converted 3d model was saved at {0}\n".format(result_filename))

        else:
            print("Model file converter failed!\n")
            sys.exit(0)
        
        

    else:

        print("The input file is missing or not readable!\n")

        print("Exiting the program...")

        sys.exit(0)

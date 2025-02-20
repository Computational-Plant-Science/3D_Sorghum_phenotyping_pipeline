#stockpile_volume.py
"""
Version: 1.5

Summary: Compute traits from a 3D model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 stockpile_volume.py -i /input_path/ 


"""



if __name__ == '__main__':
    
    
    '''
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to individual folders")
    ap.add_argument("-r", "--target_path", required = False, help = "path to target folders")
    ap.add_argument("-tq", "--type_quaternion", required = True, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")
    args = vars(ap.parse_args())
    '''
    
    pcd = o3d.io.read_point_cloud("data/stockpile.ply")

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

    o3d.visualization.draw_geometries([pcd, axes])
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=10000)[a, b, c, d] = plane_model
                                         
	plane_pcd = pcd.select_by_index(inliers)
	
	plane_pcd.paint_uniform_color([1.0, 0, 0])
	
	stockpile_pcd = pcd.select_by_index(inliers, invert=True)
	
	stockpile_pcd.paint_uniform_color([0, 0, 1.0])o3d.visualization.draw_geometries([plane_pcd, stockpile_pcd, axes])
	
	
   

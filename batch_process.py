"""
Version: 1.5

Summary: Compute traits from a 3D model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 batch_process.py -i /input_path/ 


INPUT:

    3D Sorghum model

OUTPUT:

    3D Sorghum model, aligned along Z direction in 3D coordinates
    
    Excel file contains traits computation results



PARAMETERS:

    ("-i", "--input", dest = "input", required = True, type = str, help = "full path to 3D model file")
    ("-o", "--output_path", dest = "output_path", required = False, type = str, help = "result path")
    ("--n_plane", dest = "n_plane", type = int, required = False, default = 5,  help = "Number of planes to segment the 3d model along Z direction")
    ("--slicing_ratio", dest = "slicing_ratio", type = float, required = False, default = 0.10, help = "ratio of slicing the model from the bottom")
    ("--adjustment", dest = "adjustment", type = int, required = False, default = 0, help = "adjust model manually or automatically, 0: automatically, 1: manually")
    ("--visualize", dest = "visualize", type = int, required = False, default = 0, help = "Display model or not, default as no due to headless display in cluster")


"""

import subprocess, os, glob
import sys
import argparse

import pathlib

import shutil


# execute script inside program
def execute_script(cmd_line):
    
    try:
        #print(cmd_line)
        #os.system(cmd_line)

        process = subprocess.getoutput(cmd_line)
        
        print(process)
        
        #process = subprocess.Popen(cmd_line, shell = True, stdout = subprocess.PIPE)
        #process.wait()
        #print (process.communicate())
        
    except OSError:
        
        print("Failed ...!\n")





# execute pipeline scripts in order
def model_analysis_pipeline(file_path, filename, basename, result_path):
    
    
    # step 0  python3 /opt/code/model_clean_3D.py -i ~/example/sp_test/test.ply -o ~/example/sp_test/results/ --outlier_ratio 0.1
    print("Step 0: Statistical outlier removal for 3d point cloud...\n")

    format_convert = "python3 model_clean_3D.py -i " + file_path + filename + " -o " + result_path + " --std_ratio " + str(std_ratio) 
    
    print(format_convert)
    
    execute_script(format_convert)
    
    
    # step 1  python3 /opt/code/model_alignment.py -i ~/example/test.ply  -o ~/example/result/ --n_plane 5 --slicing_ratio 0.1 --adjustment 0
    print("Step 1: Transform point cloud model to its rotation center and align its upright orientation with Z direction...\n")

    format_convert = "python3 model_alignment.py -i " + result_path + basename + "_cleaned.ply " + " -o " + result_path + " --n_plane " + str(n_plane) + " --slicing_ratio " + str(slicing_ratio) + " --adjustment " + str(adjustment)
    
    print(format_convert)
    
    execute_script(format_convert)
    
    
    
    # step 2 python3 /opt/code/model_measurement.py -i ~/example/result/test_aligned.ply  -o ~/example/result/ --n_plane 5
    print("Step 2: Compute 3D traits from the aligned 3D point cloud model...\n")

    traits_computation = "python3 model_measurement.py -i " + result_path + basename + "_cleaned_aligned.ply " + " -o " + result_path + " --n_plane " + str(n_plane)
    
    print(traits_computation)
    
    execute_script(traits_computation)
    

    
'''
# parallel processing of folders for local test only
def parallel_folders(subfolder_path):

    folder_name = os.path.basename(subfolder_path) 

    subfolder_path = os.path.join(subfolder_path, '')

    m_file = subfolder_path + folder_name + '.' + ext

    print("Processing 3d model point cloud file '{}'...\n".format(m_file))

    (filename, basename) = get_fname(m_file)

    #print("Processing 3d model point cloud file '{}'...\n".format(filename))

    #print("Processing 3d model point cloud file basename '{}'...\n".format(basename))

    model_analysis_pipeline(subfolder_path, filename, basename, subfolder_path)
'''



# get file information from the file path using os for python 2.7
def get_fname(file_full_path):
    
    abs_path = os.path.abspath(file_full_path)

    filename= os.path.basename(abs_path)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    return filename, base_name




# get sub folders from a inout path for local test only
def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders




# get file information from the file path using pathon 3
def get_file_info(file_full_path):

    p = pathlib.Path(file_full_path)
    
    filename = p.name
    
    basename = p.stem


    file_path = p.parent.absolute()
    
    file_path = os.path.join(file_path, '')
    
    return file_path, filename, basename


# execute pipeline scripts in order
def file_move(source_file_path, target_file_path):
    
    #file_path = folder_name + '_quaternion.xlsx'
    
    #file_path = folder_name + '_seg.png'
    
    batch_cmd = "cp " + source_file_path + " " + target_file_path
    
    print(batch_cmd)
    
    execute_script(batch_cmd)



# generate foloder to store the output results
def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #printpath + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        shutil.rmtree(path)
        os.makedirs(path)
        print("{} path exists!\n".format(path))
        return False
        



if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest = "input", type = str, required = True, help = "full path to 3D model file")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default='ply', help = "file type")
    ap.add_argument("--std_ratio", required = False, type = float, default = 5.0, help = "outlier remove ratio, small number = aggresive")
    ap.add_argument( "--n_plane", dest = "n_plane", type = int, required = False, default = 5,  help = "Number of planes to segment the 3d model along Z direction")
    ap.add_argument( "--slicing_ratio", dest = "slicing_ratio", type = float, required = False, default = 0.10, help = "ratio of slicing the model from the bottom")
    ap.add_argument( "--adjustment", dest = "adjustment", type = int, required = False, default = 0, help = "adjust model manually or automatically, 0: automatically, 1: manually")
    args = vars(ap.parse_args())
    



    '''
    ######################################################################################
    # create individual folders and move models
    # setting path to model file
    file_path = args["input"]

    ext = args['filetype'].split(',') if 'filetype' in args else []

    patterns = [os.path.join(file_path, f"*.{p}") for p in ext]

    model_List = [f for fs in [glob.glob(pattern) for pattern in patterns] for f in fs]



    # load input model files

    if len(model_List) > 0:

        #print("Model files in input folder: '{}'\n".format(model_List))

        for model_id, model in enumerate(model_List):
        
            # get input file path, name, base name.
            (file_path, filename, basename) = get_file_info(model)
        
            #print("Input 3d model point cloud file path: {}, filename: {}\n".format(file_path, filename))
            
            
            mkpath = os.path.dirname(file_path) + '/' + basename
            mkdir(mkpath)
            output_path = mkpath + '/'
            
            
            source_file = os.path.join(file_path,'') + filename

            target_file = os.path.join(output_path, '') + basename + '.ply'
            
            print("source_file: {}".format(source_file))
            print("target_file: {}\n".format(target_file))
            
            file_move(source_file, target_file)
        
        

    else:
        print("3D model file does not exist")
        sys.exit()


    ##################################################################################
    

    '''
    ####################################################################################
    # local test loop version
    
     # get input file path, name, base name.
    #(file_path, filename, basename) = get_file_info(args["input"])
    
    std_ratio = args["std_ratio"]

    # number of slices for cross-section
    n_plane = args['n_plane']

    slicing_ratio = args["slicing_ratio"]

    adjustment = args["adjustment"]
    
    
    file_path = args["input"]
    
    subfolders = fast_scandir(file_path)
    
    for subfolder_id, subfolder_path in enumerate(subfolders):
    
        
        folder_name = os.path.basename(subfolder_path) 
        
        subfolder_path = os.path.join(subfolder_path, '')
        
        m_file = subfolder_path + folder_name + '.ply' 
        
        #print("Processing 3d model point cloud file '{}'...\n".format(m_file))
        
        (filename, basename) = get_fname(m_file)

        
        print("Current sub folder path '{}'...\n".format(subfolder_path))

        print("Processing 3d model point cloud file '{}'...\n".format(filename))
        
        #print("Processing 3d model point cloud file basename '{}'...\n".format(basename))
        
        model_analysis_pipeline(subfolder_path, filename, basename, subfolder_path)
    
    ##################################################################################
    
    

    

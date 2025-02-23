# 3D Sorghum phenotyping

Pipeline: Traits computation from 3D Sorghum root models 




![Optional Text](../main/media/3A10.png)

Example of 3D Sorghum root models  (genotype 3A10)

![Optional Text](../main/media/5A22.png)

Example of 3D Sorghum root models (genotype 5A22)


![Optional Text](../main/media/Sorghum_demo.gif)

One sample trait computed from 3D Sorghum point cloud model

Angle of simplified root structure curve line


![Optional Text](../main/media/5A22_R1_traits.png)

Computed traits illustration 




## Input


3D root models (*.ply) in Polygon File Format or the Stanford Triangle Format. 

computed from Computational-Plant-Science / 3D_model_reconstruction_demo
(https://github.com/Computational-Plant-Science/3D_model_reconstruction_demo)


## Output

trait.xlsx   Excel format file contains computed root traits




## Requirements

[Docker](https://www.docker.com/) is required to run this project in a Linux environment.

Install Docker Engine (https://docs.docker.com/engine/install/)



## Usage


We suggest to run the pipeline inside a docker container, 

The Docker container allows you to package up your application(s) and deliver them to the cloud without any dependencies. It is a portable computing environment. It contains everything an application needs to run, from binaries to dependencies to configuration files.


There are two ways to run the pipeline inside a docker container, 

One was is to build a docker based on the docker recipe file inside the GitHub repository. In our case, please follow step 1 and step 3. 

Antoher way is to download prebuild docker image directly from Docker hub. In our case, please follow step 2 and step 3. 


1. Build docker image on your PC under linux environment
```shell

git clone https://github.com/Computational-Plant-Science/3D_Sorghum_phenotyping_pipeline.git

docker build -t 3d-model-traits -f Dockerfile .
```
2. Download prebuild docker image directly from Docker hub, without building docker image on your local PC 
```shell
docker pull computationalplantscience/3d_sorghum_phenotyping
```
3. Run the pipeline inside the docker container 

link your test 3D model path (e.g. '/home/test/test.ply', $path_to_your_3D_model = /home/test, $your_3D_model_name.ply = test.ply)to the /srv/test/ path inside the docker container
 ```shell
docker run -v /$path_to_your_3D_model:/srv/test -it 3d-model-traits

or 

docker run -v /$path_to_your_3D_model:/srv/test -it computationalplantscience/3d_sorghum_phenotyping

```

4. Run the pipeline inside the docker container
```shell
python3 /opt/code/pipeline.py -i /srv/test/test.ply -o /srv/test/result/

```
  



# Author

Suxing Liu (suxingliu@gmail.com)


## Other contributions

Docker container was maintained and deployed to [PlantIT](https://portnoy.cyverse.org) by Suxing Liu (suxingliu@gmail.com), Wes Bonelli (wbonelli@ucar.edu).


# License
GNU Public License



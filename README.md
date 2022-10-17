# ImageSegmentation

Repository to for CS-155 Algorithmic Segmentation and CS-156 Implement Advanced Segmentation Solution.  

## Input data  
Get the input folder from the link below, unzip and put the folder ("input/") in the "test" folder  after git cloning (so "test/input/"). This is necessary as the data is too large to upload to GitHub.  
URL: https://flinders-my.sharepoint.com/:u:/g/personal/duns0089_flinders_edu_au/EesXhHx6lnVMiAvKEGI2UlUBbLlfCROKaK_QtE_k2UQDSw?e=sdEmWg  

## System
### Test Computer Specifications:
![test comupter specifications](documentation/diagrams/test_system_specifications.png)

The Meson Build System is also required, v0.61.2.  
Python 3.10.4.  

## Toolkit Architecture
See References to see sources of each algorithm  
![toolkit architecture](documentation/diagrams/architecture.png)

## Running the toolkit
There a number of ways to run the toolkit. The easiest way is to use the *./run-test.sh* file. This file contains a number of commented out lines that correspond to the the files that will be segmented. The -r option means that minimal spanning tree segmentations will occur and need reconstruction greyscale image. The non -r option requires a uintt_8 file with the material compounds look up table specified as well.  

## Test Results
These are the results from the thesis located in documention/. 

![ccl results](documentation/diagrams/ccl_segmentation_times.png)

![mst results](documentation/diagrams/mst_segmentation_times.png)

## References
Arbitary CCL: From https://ieeexplore.ieee.org/document/87344/ but algorithm better explained in
https://en.wikipedia.org/wiki/Connected-component_labeling#One_component_at_a_time  
Playne-Equivalence Parallel CUDA CCL: https://github.com/DanielPlayne/playne-equivalence-algorithm  
FelzenszwalbMST by Motwani: https://iammohitm.github.io/Graph-Based-Image-Segmentation/  
Ganin Parallel CUDA MST: https://developer.download.nvidia.com/GTC/PDF/GTC2012/Posters/P0496_Efficient_Segmentation_Trees_by_Ganin.pdf   
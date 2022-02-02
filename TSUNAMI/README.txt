Summary
====================================================================

This dataset contains the image subset named “TSUNAMI”. We make them 
publicly available for the researchers who are interested in the 
problem of the image-based detection of temporal scene changes. 
Although we own its copyright, you can freely use it for research 
purposes. We request that you cite the following paper if you publish 
research results utilizing these data:

Ken Sakurada and Takayuki Okatani, Change Detection from a Street 
Image Pair using CNN Features and Superpixel Segmentation, 
Proc. British Machine Vision Conference (BMVC), 2015.

URL of the dataset:
http://www.vision.is.tohoku.ac.jp/us/download/

Description
====================================================================

"TSUNAMI" consists of one hundred panoramic image pairs of scenes in 
tsunami-damaged areas of Japan. The size of these images is 
224 × 1024 pixels.

For each of them, we hand-labeled the ground truth of scene changes. 
It is given in the form of binary image of the same size as the input 
pair of images. The binary value at each pixel indicates that a change 
has occurred at the corresponding scene point on the paired images. 
We defined the scene changes to be detected as 2D changes of surfaces 
of objects (e.g., changes of the advertising board) and 3D, structural 
changes (e.g., emergence/vanishing of buildings and cars). The changes 
due to differences in illumination and photographing condition and 
those of the sky and the ground are excluded, such as changes due to 
specular reflection on building windows and changes of cloud and 
signs on the road surface.
 

Ground truth
====================================================================

All image pairs have ground truths of temporal scene changes, which 
are manually obtained by ourselves. They are stored in "ground_truth/*.bmp."



Directory structure
====================================================================

Directory structure

TSUNAMI
| —-README.txt
| --t0 // *.jpg
| --t1 // *.jpg
| --ground_truth // *.bmp
    

We welcome your questions, comments and suggestions. Please send them to 
sakurada"at"vision.is.tohoku.ac.jp or okatani"at"vision.is.tohoku.ac.jp

Ken Sakurada and Takayuki Okatani
Tohoku University, Japan
August 2015

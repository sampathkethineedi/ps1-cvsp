# PS1 - Visual SLAM (Computer Vision)

This repositry is majorly based on the depth estimation GitHub repository 'monodepth2' by nianticlabs. Link to original repo:- https://github.com/nianticlabs/monodepth2 . 
For perspective transformation i referred to the following GitHub repo. :-https://github.com/darylclimb/cvml_project/tree/master/projections/inverse_projection
I tried to analyse the different models proposed by them and drawn comparision between them in the documentation. An accuracy based comparision is available on the original Repo.

I've added 2 files 'geometry_utils.py' and 'finalDemo.py' with some references to original repository. The 'finalDemo.py' can be used to test a single mono image to estimate the depth image ,setting the relative range of depth and plotting the 3-D point clouds using open3D library.

## Testing your image

The finalDemo.py file can be used to test and get 3-D visualization of the single image. It takes 2 arguments that are image path and model name.
Choices for model are the same as the original pre-trained models from monodepth2 : 
1.mono_640x192
2.stereo_640x192
3.mono_1024x320
4.stereo_1024x320

Running the file:-
Use the following command to run the finalDemo.py file:
```shell
python finalDemo.py --image_path assets/s3.png --model_name mono_640x192
```

The output of the file will be a depth vs rgb image comparision and Open3D 3-D plot of the 2-D image.

## Environment Specifications

1. conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
2. pip install tensorboardX==1.4
3. conda install opencv=3.3.1
4. pip install open3d





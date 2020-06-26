import cv2
import os
import sys
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import time
import torch
from torchvision import transforms
import scipy.misc as smp
import argparse
from geometry_utils import *
import open3d as o3d

import networks
from utils import download_model_if_doesnt_exist

def get_args():
	parse = argparse.ArgumentParser()

	parse.add_argument('--image_path',type=str,help= 'insert image path',required= True)
	parse.add_argument('--model_name',type=str,help='model name',required=True)
	
	return parse.parse_args()

def disp_to_depth(disp,min_depth,max_depth):
	min_disp = 1/max_depth
	max_disp = 1/min_depth
	scaled_disp = min_disp + (max_disp - min_disp)*disp
	dis = 1/scaled_disp
	return scaled_disp,dis

def plot(rgb,depth):
	plt.figure(figsize=(10, 10))
	plt.subplot(211)
	plt.imshow(rgb)
	plt.title("RGM IMAGE", fontsize=22)
	plt.axis('off')

	plt.subplot(212)
	plt.imshow(depth)
	plt.title("Depth image", fontsize=22)
	plt.axis('off')
	plt.show()

def depth_Estimation(args):
	model_name = args.model_name
	#Setting up the network
	print("Loading model....")
	download_model_if_doesnt_exist(model_name)
	encoder_path = os.path.join("models", model_name, "encoder.pth")
	depth_decoder_path = os.path.join("models", model_name, "depth.pth")

	# LOADING PRETRAINED MODEL
	encoder = networks.ResnetEncoder(18, False)
	depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

	loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
	filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
	encoder.load_state_dict(filtered_dict_enc)

	loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
	depth_decoder.load_state_dict(loaded_dict)

	encoder.eval()
	depth_decoder.eval();

	#Loading image
	print("Loading image....")
	image_path = args.image_path
	input_image = pil.open(image_path).convert('RGB')
	original_width, original_height = input_image.size
	feed_height = loaded_dict_enc['height']
	feed_width = loaded_dict_enc['width']
	input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

	input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
	input_npy = input_image_pytorch.squeeze().cpu().numpy()


	#prediction of disparity image
	with torch.no_grad():
		features = encoder(input_image_pytorch)
		outputs = depth_decoder(features)
		disp = outputs[("disp", 0)]

	 #Scaling for given resolution
	disp_resized = torch.nn.functional.interpolate(disp,
	(original_height, original_width), mode="bilinear", align_corners=False) # interpolate the values in to fit the given resolution of the image

	disp_resized_np = disp_resized.squeeze().cpu().numpy() # Converting tensor in pytorch to numpy array
	print("resized disp" + str(disp_resized_np.shape))
	print("Range of Depth in image")
	scaled,dep = disp_to_depth(disp_resized_np,0.1,1000) # resizing the depth from 0.1 to 100 units
	print("min->"+str(dep.min())+"mx->"+str(dep.max()))
	#Preview of the rgb and Depth images
	rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
	depth = dep.reshape((rgb.shape[0],rgb.shape[1]),order='C')
	plot(rgb,depth)

	return rgb,depth
def Perspective_transformation(rgb,depth,args):
	height,width,_ = rgb.shape

	K = intrinsic_from_fov(height, width, 90)  # +- 45 degrees
	K_inv = np.linalg.inv(K)
	pixel_coords = pixel_coord_np(width, height)  # [3, npoints]

	# Apply back-projection: K_inv @ pixels * depth
	cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()

	#Select the range till u want to project points in 3-D
	limit_of_depth = input("Limit of the depth to view\n") # upper limit of depth
	limit_of_depth = float(limit_of_depth)
	cam_coords = cam_coords[:, np.where(cam_coords[2] <= limit_of_depth)[0]]

	return cam_coords

def Visualization_3D(rgb,cam_coords):
	col = rgb.copy()
	rgb_t =col.reshape((rgb.shape[0]*rgb.shape[1],3))
	pcd_cam = o3d.geometry.PointCloud()
	pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
	pcd_cam.colors = o3d.utility.Vector3dVector(rgb_t.astype(np.float) / 255.0)
	# Flip it, otherwise the pointcloud will be upside down
	pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd_cam])

if __name__ == '__main__':
	args = get_args()
	rgb,depth = depth_Estimation(args)
	coord3d = Perspective_transformation(rgb,depth,args)

	Visualization_3D(rgb,coord3d)

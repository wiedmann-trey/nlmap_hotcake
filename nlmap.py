import configparser
import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
from PIL import Image
import clip
import torch

from nlmap_utils import get_best_clip_vild_dirs
from spot_utils.utils import pixel_to_vision_frame, pixel_to_vision_frame_depth_provided, arm_object_grasp, open_gripper
from spot_utils.generate_pointcloud import make_pointcloud
from vild.vild_utils import visualize_boxes_and_labels_on_image_array, plot_mask
import matplotlib.pyplot as plt
from matplotlib import patches

import cv2

import open3d as o3d

from bosdyn.client.image import ImageClient

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import geometry_pb2
from bosdyn.api import basic_command_pb2

from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                        block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from spot_utils.move_spot_to import move_to

import tensorflow.compat.v1 as tf
from vild.vild_utils import build_text_embedding

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

from sklearn.cluster import DBSCAN
import copy

import random

import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

from pathlib import Path
import PIL.Image
import shutil

FEAT_SIZE = 512

class NLMap():
	def __init__(self,config_path="./configs/example.ini"):
		###########################################################################################################
		######### Initialization

		if not os.path.isfile(config_path):
			raise Exception(f"config_path {config_path} has no config file")
		### Extract config file 
		self.config = configparser.ConfigParser()
		self.config.sections()
		self.config.read(config_path)

		### device setting
		device = "cuda" if torch.cuda.is_available() else "cpu"

		### CLIP models set to none by default
		self.clip_model = None
		self.clip_preprocess = None

		### Robot initializaton
		if self.config["robot"].getboolean("use_robot"):
			self.sdk = bosdyn.client.create_standard_sdk('NLMapSpot')
			self.robot = self.sdk.create_robot(self.config["robot"]["hostname"])
			bosdyn.client.util.authenticate(self.robot)

			# Time sync is necessary so that time-based filter requests can be converted
			self.robot.time_sync.wait_for_sync()

			assert not self.robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
			                                "such as the estop SDK example, to configure E-Stop."

			self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

			self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
			self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
			self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
			self.robot_command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

			#We only use the hand color
			self.sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']

		### Values set based on config parameters
		self.data_dir_path = f"{self.config['paths']['data_dir_root']}/{self.config['dir_names']['data']}"
		self.figs_dir_path = f"{self.config['paths']['figs_dir']}/{self.config['dir_names']['data']}"
		if ((self.config["viz"].getboolean("save_whole_boxes") or self.config["viz"].getboolean("save_anno_boxes")) and not os.path.isdir(self.figs_dir_path)):
			os.mkdir(self.figs_dir_path)

		### Load pose data
		## this sets up the directory variables with the pose information
		if self.config["pose"].getboolean("use_pose"): ##### TODO read from .xml
			pose_path = f"{self.data_dir_path}/{self.config['file_names']['pose']}"
			try:
				self.pose_dir = pickle.load(open(pose_path,"rb")) 
			except:
				raise Exception(f"use_pose is true but no pose data found at {pose_path}")

		### Pointcloud initialization
		if self.config["pointcloud"].getboolean("use_pointcloud"):
			pointcloud_path = f"{self.data_dir_path}/{self.config['file_names']['pointcloud']}"
			if os.path.isfile(pointcloud_path):
				self.pcd = o3d.io.read_point_cloud(pointcloud_path)
			else:
				#raise Exception(f"use_pointcloud is true but {pointcloud_path} does not exist. Implement GENERATE POINTCLOUD")
				self.pcd = make_pointcloud(data_path=f"{self.data_dir_path}/",pose_data_fname=self.config["file_names"]["pose"], pointcloud_fname=self.config["file_names"]["pointcloud"])

		### Text initialization
		self.category_names = [x.strip() for x in self.config["text"]["category_name_string"].split(';')]
		self.categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(self.category_names)]
		self.category_indices = {cat['id']: cat for cat in self.categories}



		### Cache path
		self.cache_path = f"{self.config['paths']['cache_dir']}/{self.config['dir_names']['data']}"
	
		### Compute text embeddings with CLIP 
		self.compute_text_embeddings()
		
		### Image initialization
		self.image_names = os.listdir(self.data_dir_path)
		self.image_names = [image_name for image_name in self.image_names if "color" in image_name]

		###########################################################################################################
		######### Image embeddings

		### Load cached image embeddings if they exist and are to be used, or make them otherwise
		self.cache_image_exists = os.path.isfile(f"{self.config['paths']['cache_dir']}/{self.config['dir_names']['data']}_images_vild")

		## huda TODO populate the ground truth dictionary (class variable)

		columns = ['position_x', 'position_y','position_z', "bounding_box_y1","bounding_box_x1",  "bounding_box_y2","bounding_box_x2",'image_index',"pred_anno_idx", 'ground_truth_anno_idx']
		columns_emb_name = [f"vild_embedding_{i}" for i in range(FEAT_SIZE)]
		columns = np.append(columns, columns_emb_name)
		
		if self.config["cache"].getboolean("images") and self.cache_image_exists: #if image cache should be used and it exists, load it in
			self.image2vectorvild_dir = pickle.load(open(f"{self.cache_path}_images_vild","rb"))
			self.image2vectorclip_dir = pickle.load(open(f"{self.cache_path}_images_clip","rb"))
			df = pd.read_csv(f"{self.cache_path}_embeddings.csv",  dtype={'image_index': 'object'}, index_col=0)
			cluster_count_df = pd.read_csv(f"{self.cache_path}_cluster.csv", index_col=0)

			if not self.config["our_method"].getboolean("use_our_method"):
				self.topk_vild_dir= pickle.load(open(f"{self.cache_path}_topk_vild","rb"))
				self.topk_clip_dir = pickle.load(open(f"{self.cache_path}_topk_clip","rb"))
		else: #make image embeddings (either because you're not using cache, or because you don't have cache)
			self.priority_queue_clip_dir = defaultdict(PriorityQueue) #keys will be category names. The priority will be negative score (since lowest gets dequeue) and items be image, anno_idx, and crop
			self.priority_queue_vild_dir = defaultdict(PriorityQueue) #keys will be category names. The priority will be negative score (since lowest gets dequeue) and items be image, anno_idx, and crop

			# Load ViLD model
			from vild.vild_utils import extract_roi_vild, paste_instance_masks

			self.session = tf.Session(graph=tf.Graph())
			_ = tf.saved_model.loader.load(self.session, ['serve'], self.config["paths"]["vild_dir"])

			## load CLIP model
			if self.config["our_method"].getboolean("use_clip"):
				self.clip_model, self.clip_preprocess = clip.load(self.config["clip"]["model"])

			params = self.config["vild"].getint("max_boxes_to_draw"),  self.config["vild"].getfloat("nms_threshold"),  self.config["vild"].getfloat("min_rpn_score_thresh"),  self.config["vild"].getfloat("min_box_area")

			print("Computing image embeddings")
			self.image2vectorvild_dir = {}
			self.image2vectorclip_dir = {}
			embedding_points = []
			count = 0 
			
			df = pd.DataFrame()
			df = pd.DataFrame(columns = columns)
			self.image_names = sorted(self.image_names)
			for image_name in tqdm(self.image_names):
				count+=1
				img_index = int(image_name.split("_")[1][0])
				torch.cuda.empty_cache()
				print(image_name)
				image_path = f"{self.data_dir_path}/{image_name}"

				image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes  = extract_roi_vild(image_path,self.session,params)
				self.image2vectorvild_dir[image_name] = [image,image_height,image_width,valid_indices,detection_roi_scores,detection_boxes,detection_masks,detection_visual_feat,rescaled_detection_boxes]
			
				### We only compute CLIP embeddings for vild crops that have highest score
				### Compute detection scores, and rank results

				## if our method is on, then just use all the indicies in scores_all without ranking them
				
				if self.config["our_method"].getboolean("use_our_method"):
					indices = list(range(self.config["vild"].getint("max_boxes_to_draw")))
				else:
					raw_scores = detection_visual_feat.dot(self.text_features.T)
					

					if self.config["vild"].getboolean("use_softmax"):
						scores_all = softmax(temperature * raw_scores, axis=-1)
					else:
						scores_all = raw_scores

					indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores

				## saving whole image visualizations with bounding boxes
				ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
				processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
				segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

				overall_fig_size = [float(x) for x in self.config["viz"]["overall_fig_size"].split(",")]

				if self.config["viz"].getboolean("boxes") or self.config["viz"].getboolean("save_whole_boxes"):
					if len(indices) == 0:
						if self.config["viz"].getboolean("boxes"):
							display_image(np.array(image), size=overall_fig_size)
						print('ViLD does not detect anything belong to the given category')

					else:
						image_with_detections = visualize_boxes_and_labels_on_image_array(
							np.array(image),
							rescaled_detection_boxes[indices],
							valid_indices[:self.config["vild"].getint("max_boxes_to_draw")][indices],
							detection_roi_scores[indices],
							self.category_indices,
							instance_masks=segmentations[indices],
							use_normalized_coordinates=False,
							max_boxes_to_draw=self.config["vild"].getint("max_boxes_to_draw"),
							min_score_thresh=self.config["vild"].getfloat("min_rpn_score_thresh"),
							skip_scores=False,
							skip_labels=True)


					plt.figure(figsize=overall_fig_size)
					plt.imshow(image_with_detections)
					plt.axis('off')
					plt.title('Detected objects and RPN scores')
					if self.config["viz"].getboolean("save_whole_boxes"):
						plt.savefig(f"{self.figs_dir_path}/{image_name}_whole.jpg", bbox_inches='tight')
					if self.config["viz"].getboolean("boxes"):
						plt.show()
					plt.close()
				else:
					if len(indices) == 0:
						print(f'ViLD does not detect anything belong to the given category for {image_name}')					
					
				raw_image = np.array(image)
				n_boxes = rescaled_detection_boxes.shape[0]

				### image2vectorclip_dir[image_name] is a directory with annotations (crops) as keys
				self.image2vectorclip_dir[image_name] = {}

				### Go through the top crops (baseline) or all crops (our_method)
				for anno_idx in indices[0:int(n_boxes)]:
					# continue
					rpn_score = detection_roi_scores[anno_idx]
					bbox = rescaled_detection_boxes[anno_idx]
					if not self.config["our_method"].getboolean("use_our_method"):
						scores = scores_all[anno_idx]

					y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
					crop = np.copy(raw_image[y1:y2, x1:x2, :])

					### Add crop to priority queue for ranking scores for VILD
						
					### Run CLIP vision model on crop
					crop_pil = Image.fromarray(crop)

					#if (use cache and cache does not exist) or (cache image does not exist), process the data
					if ((self.config["cache"].getboolean("images") and not self.cache_image_exists) or (not self.cache_image_exists)):
						crop_fname = f"{self.cache_path}_{self.config['dir_names']['data']}_{image_name}_crop_{anno_idx}.jpeg"
						crop_pil.save(crop_fname)
						crop_back = Image.open(crop_fname)
						if self.config["our_method"].getboolean("use_clip"):
							crop_processed = self.clip_preprocess(crop_back).unsqueeze(0).to(device)
							clip_image_features = self.clip_model.encode_image(crop_processed)

							self.image2vectorclip_dir[image_name][anno_idx] = clip_image_features

					if self.config["our_method"].getboolean("use_clip"):
						clip_image_features = self.image2vectorclip_dir[image_name][anno_idx]

						#Normalize clip_image_features before taking dot product with normalized text features
						clip_image_features = clip_image_features / clip_image_features.norm(dim=1, keepdim=True)
						clip_image_features = clip_image_features.cpu().detach().numpy()
					if not self.config["our_method"].getboolean("use_our_method"):	
						if self.config["our_method"].getboolean("use_clip"):
							clip_scores = clip_image_features.dot(self.text_features.T)
							### Add crop to priority queue for ranking scores for CLIP
						for idx, category_name in enumerate(self.category_names):
							if self.config["our_method"].getboolean("use_clip"):
								
								self.priority_queue_clip_dir[category_name].put((-clip_scores[0][idx], (image_name,anno_idx,crop,ymin[anno_idx],xmin[anno_idx],ymax[anno_idx],xmax[anno_idx]))) #TODO: make this an object to more interpretable
						
							new_item = (-scores[idx], (image_name,anno_idx,crop,ymin[anno_idx],xmin[anno_idx],ymax[anno_idx],xmax[anno_idx]))
							if new_item in self.priority_queue_vild_dir[category_name].queue:
								raise Exception(f"{image_name} {anno_idx} already in queue for {category_name}")
							self.priority_queue_vild_dir[category_name].put(new_item) #TODO: make this an object to more interpretable

						self.save_anno_boxes(image_name,anno_idx, scores, raw_image, segmentations, rpn_score, crop, x1,x2,y1,y2)
					
					# 	# TODO 3d_position does not get initialized if there are no ground truths associated with that image.
					_3d_poisiton = self.extract_3d_position(image_name, x1,x2,y1,y2)
					embedding = np.append(_3d_poisiton, [y1, x1, y2, x2 ])
					embedding = np.append(embedding, [img_index])
					embedding = np.append(embedding, [crop_fname])
					embedding = np.append(embedding, [None]) # TODO add ground truth object name 
					embedding = np.append(embedding, detection_visual_feat[anno_idx])

					## order of the columns
					# ['position_x', 'position_y','position_z', 
					# "bounding_box_y1","bounding_box_x1",  "bounding_box_y2","bounding_box_x2" 
					# 'image_index',"pred_anno_idx", 
					# 'ground_truth_anno_idx', 'embedding]

					df.loc[len(df.index)] = embedding
					# TODO: fig_size_w and h are a little hardcoded, make more general?
					
				
			if self.config["cache"].getboolean("images"):
				pickle.dump(self.image2vectorvild_dir,open(f"{self.cache_path}_images_vild","wb"))
				pickle.dump(self.image2vectorclip_dir,open(f"{self.cache_path}_images_clip","wb"))

			### For priority queue, just get the top k results since PriorityQueue is not picklable
			if not self.config["our_method"].getboolean("use_our_method"):					
				self.topk_vild_dir = {}
				self.topk_clip_dir = {}
				for category_name in self.category_names:
					topk_vild_list = []
					topk_clip_list = []
					for k in range(self.config["fusion"].getint("top_k")):
						top_k_item_vild = self.priority_queue_vild_dir[category_name].get()
						top_k_item_clip = self.priority_queue_clip_dir[category_name].get()

						topk_vild_list.append(top_k_item_vild)
						topk_clip_list.append(top_k_item_clip)

					self.topk_vild_dir[category_name] = topk_vild_list
					self.topk_clip_dir[category_name] = topk_clip_list


				if self.config["cache"].getboolean("images"):
					pickle.dump(self.topk_vild_dir,open(f"{self.cache_path}_topk_vild","wb"))
					pickle.dump(self.topk_clip_dir,open(f"{self.cache_path}_topk_clip","wb"))

		print("clustering starts")
		print(df.columns)
		subset_columns = np.append(columns_emb_name, ['position_x', 'position_y','position_z'])
		objects_df = df[subset_columns] 

		if self.config['our_method']['analysis']:
		### do  clustering analysis
			neighbors = NearestNeighbors(n_neighbors=5)
			neighbors_fit = neighbors.fit(objects_df)
			distances, indices = neighbors_fit.kneighbors(objects_df)
			plt.figure(figsize=(10,6))
			distances = np.sort(distances, axis=0)
			distances = distances[:,1]
			plt.plot(distances)
			plt.title("Clusters determined by DBSCAN")
			plt.savefig(f"{self.figs_dir_path}/distances.jpg", bbox_inches='tight')
		
		clustering = DBSCAN(eps=3, min_samples=5).fit(objects_df)
		objects_df.loc[:,'cluster'] = clustering.labels_ 

		cluster_dir = Path(self.config['paths']['cluster_dir'])
		shutil.rmtree(cluster_dir)
		## reorganize image folder based on clusters
		for index, row in objects_df.iterrows():
			cluster_number = int(row["cluster"])
			if cluster_number != -1:
				crop_pil = PIL.Image.open(df.iloc[index]["pred_anno_idx"])
				cluster_dir.mkdir(parents=True, exist_ok=True)
				Path(self.config['paths']['cluster_dir']+"/"+ str(cluster_number)).mkdir(parents=True, exist_ok=True)
				filename = df.iloc[index]["pred_anno_idx"].split("/")[-1]
				crop_pil.save(self.config['paths']['cluster_dir'] + "/" + str(cluster_number)+ "/" + filename)


		cluster_count_df = objects_df.cluster.value_counts().to_frame()
		if self.config["cache"].getboolean("images"):
			print(df["image_index"].dtypes)
			df.to_csv(f"{self.cache_path}_embeddings.csv")
			cluster_count_df.to_csv(f"{self.cache_path}_cluster.csv")
		X_embedded = TSNE(n_components=2, perplexity = 3).fit_transform(objects_df)
		objects_df["x_component"]=X_embedded[:,0]
		objects_df["y_component"]=X_embedded[:,1]
		objects_df["image_index"] = df["image_index"]
		objects_df["pred_anno_idx"] = df["pred_anno_idx"]

		fig = px.scatter(objects_df, x="x_component", y="y_component", hover_data=["cluster", "pred_anno_idx"], color = "image_index")
		fig.update_layout(
			height=800)
		fig.write_html(f"{self.figs_dir_path}/clustering.html")


		print("done with storing cache of embedding is done")	

	def compute_text_embeddings(self):
		self.cache_text_exists = os.path.isfile(f"{self.cache_path}_text")

		#if text cache should be used and it exists, load it
		if self.config["cache"].getboolean("text") and self.cache_text_exists:
			self.text_features = pickle.load(open(f"{self.cache_path}_text","rb"))
		else: #build new text embeddings

			#load CLIP model
			self.clip_model, self.clip_preprocess = clip.load(self.config["clip"]["model"])

			self.text_features = build_text_embedding(self.categories,self.clip_model,self.clip_preprocess,prompt_engineering=self.config["text"].getboolean("prompt_engineering"))

			if self.config["cache"].getboolean("text"): #save the text cache
				pickle.dump(self.text_features,open(f"{self.cache_path}_text","wb"))

	def save_anno_boxes(self,image_name, anno_idx, scores, raw_image, segmentations, rpn_score, crop, x1,x2,y1,y2):
		fig_size_w = 35
		fig_size_h = min(max(5, int(len(self.category_names) / 2.5) ), 10)

		if self.config["viz"].getboolean("boxes") or self.config["viz"].getboolean("save_anno_boxes"):
			img_w_mask = plot_mask(self.config["viz"]["mask_color"], self.config["viz"].getfloat("alpha"), raw_image, segmentations[anno_idx])
			crop_w_mask = img_w_mask[y1:y2, x1:x2, :]

			fig, axs = plt.subplots(1, 4, figsize=(fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

			# Draw bounding box.
			rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=self.config["viz"].getfloat("line_thickness"), edgecolor='r', facecolor='none')
			axs[0].add_patch(rect)

			axs[0].set_xticks([])
			axs[0].set_yticks([])
			axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')

			axs[0].imshow(raw_image)

			# Draw image in a cropped region.
			axs[1].set_xticks([])
			axs[1].set_yticks([])

			axs[1].set_title(f'predicted: {self.category_names[np.argmax(scores)]}')

			axs[1].imshow(crop)

			# Draw segmentation inside a cropped region.
			axs[2].set_xticks([])
			axs[2].set_yticks([])
			axs[2].set_title('mask')

			axs[2].imshow(crop_w_mask)

			# Draw category scores.
			fontsize = max(min(fig_size_h / float(len(self.category_names)) * 45, 20), 8)
			for cat_idx in range(len(self.category_names)):
				axs[3].barh(cat_idx, scores[cat_idx], 
							color='orange' if scores[cat_idx] == max(scores) else 'blue')
			axs[3].invert_yaxis()
			axs[3].set_axisbelow(True)
			axs[3].set_xlim(0, 1)
			plt.xlabel("confidence score")
			axs[3].set_yticks(range(len(self.category_names)))
			axs[3].set_yticklabels(self.category_names, fontdict={
				'fontsize': fontsize})

			if self.config["viz"].getboolean("save_anno_boxes"):
				plt.savefig(f"{self.figs_dir_path}/{image_name}_anno_{anno_idx}.jpg", bbox_inches='tight')
			if self.config["viz"].getboolean("boxes"):
				plt.show()
			plt.close()



	def viz_pointcloud(self):
		o3d.visualization.draw_geometries([self.pcd])

	def viz_top_k(self,viz_2d=True,viz_pointcloud=True):
		for category_name in self.category_names:
			print(f"category: {category_name}")
			top_axes = []

			if viz_2d:
				fig, axs = plt.subplots(2, self.config["fusion"].getint("top_k"))
				plt.suptitle(f"Query: {category_name}")

			for k in range(self.config["fusion"].getint("top_k")):
				top_k_item_vild = self.topk_vild_dir[category_name][k]
				top_k_item_clip = self.topk_clip_dir[category_name][k]

				if viz_2d:

					axs[0, k].set_title(f"ViLD score {top_k_item_vild[0]*-1:.3f}")
					axs[0, k].imshow(top_k_item_vild[1][2])

					axs[1, k].set_title(f"CLIP score {top_k_item_clip[0]*-1:.3f}")
					axs[1, k].imshow(top_k_item_clip[1][2])
				
				#### Point cloud stuff
				#### Just show CLIP for now!
				file_num = int(top_k_item_clip[1][0].split("_")[1].split(".")[0])  ########## TODO this is the pose calculation
				depth_img = pickle.load(open(f"{self.data_dir_path}/depth_{str(file_num)}","rb")) ## TODO: change this for the new dataset
				rotation_matrix = self.pose_dir[file_num]['rotation_matrix']
				position = self.pose_dir[file_num]['position'] ############# TODO read from xml
				## have if else for the code below. add a config variable

				ymin, xmin, ymax, xmax = top_k_item_clip[1][3:]

				center_y = int((ymin + ymax)/2.0)
				center_x = int((xmin + xmax)/2.0)

				transformed_point,bad_point = pixel_to_vision_frame(center_y,center_x,depth_img,rotation_matrix,position)
				side_pointx,_ = pixel_to_vision_frame_depth_provided(center_y,xmax,depth_img[center_y,center_x],rotation_matrix,position)
				side_pointy,_ = pixel_to_vision_frame_depth_provided(ymax,center_x,depth_img[center_y,center_x],rotation_matrix,position)

				#TODO: what should bb_size be for z? Right now, just making it same as x. Also needs to be axis aligned
				bb_sizex = np.linalg.norm(transformed_point-side_pointx)[0]*2
				bb_sizey = np.linalg.norm(transformed_point-side_pointy)[0]*2
				


				if bad_point:
					print(f"0 depth at the point for item {k} next bounding box")
				else:
					print(f"item {k} good inside {top_k_item_clip[1][0]}")
					print(transformed_point)
					bb = o3d.geometry.OrientedBoundingBox(center=np.array(transformed_point),R=np.array([[1,0,0],[0,1,0],[0,0,1]]), extent=np.array([bb_sizex,bb_sizex,bb_sizey]))
					bb.color = [1,0,0]
					axis_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5,origin=transformed_point)
					top_axes.append(bb)
					top_axes.append(axis_center)

			if viz_2d:
				plt.show()
			if viz_pointcloud:
				o3d.visualization.draw_geometries([self.pcd]+top_axes)

	def extract_3d_position(self, filename, xmin,xmax,ymin,ymax):
		#### Point cloud stuff
		#### Just show CLIP for now!
		file_num = int(filename.split("_")[1].split(".")[0])  ########## TODO this is the pose calculation
		depth_img = pickle.load(open(f"{self.data_dir_path}/depth_{str(file_num)}","rb")) ## TODO: change this for the new dataset
		rotation_matrix = self.pose_dir[file_num]['rotation_matrix']
		position = self.pose_dir[file_num]['position'] ############# TODO read from xml
		## have if else for the code below. add a config variable
		# ymin, xmin, ymax, xmax = top_k_item_clip[1][3:]

		center_y = int((ymin + ymax)/2.0)
		center_x = int((xmin + xmax)/2.0)

		transformed_point,bad_point = pixel_to_vision_frame(center_y,center_x,depth_img,rotation_matrix,position)
		side_pointx,_ = pixel_to_vision_frame_depth_provided(center_y,xmax,depth_img[center_y,center_x],rotation_matrix,position)
		side_pointy,_ = pixel_to_vision_frame_depth_provided(ymax,center_x,depth_img[center_y,center_x],rotation_matrix,position)

		#TODO: what should bb_size be for z? Right now, just making it same as x. Also needs to be axis aligned
		return transformed_point

	def go_to_and_pick_top_k(self, category_name):
		assert self.config["robot"].getboolean("use_robot")
		with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
			best_pose = None
			for k in range(self.config["fusion"].getint("top_k")):
				top_k_item_vild = self.topk_vild_dir[category_name][k]
				top_k_item_clip = self.topk_clip_dir[category_name][k]

				#### Point cloud stuff
				#### Just show CLIP for now!
				file_num = int(top_k_item_clip[1][0].split("_")[1].split(".")[0])
				depth_img = pickle.load(open(f"{self.data_dir_path}/depth_{str(file_num)}","rb"))
				rotation_matrix = self.pose_dir[file_num]['rotation_matrix']
				position = self.pose_dir[file_num]['position']
				## TODO wrap this in an if statement with the 2 extract pose helpers? 
				ymin, xmin, ymax, xmax = top_k_item_clip[1][3:]

				center_y = int((ymin + ymax)/2.0)
				center_x = int((xmin + xmax)/2.0)

				transformed_point,bad_point = pixel_to_vision_frame(center_y,center_x,depth_img,rotation_matrix,position)
				side_pointx,_ = pixel_to_vision_frame_depth_provided(center_y,xmax,depth_img[center_y,center_x],rotation_matrix,position)
				side_pointy,_ = pixel_to_vision_frame_depth_provided(ymax,center_x,depth_img[center_y,center_x],rotation_matrix,position)

				#TODO: what should bb_size be for z? Right now, just making it same as x. Also needs to be axis aligned
				bb_sizex = np.linalg.norm(transformed_point-side_pointx)[0]*2
				bb_sizey = np.linalg.norm(transformed_point-side_pointy)[0]*2
				

				if not bad_point:
					if type(best_pose) == type(None):
						best_pose = transformed_point

						input(f"Go to {category_name} at location {best_pose} (hit enter)")

						move_to(self.robot,self.robot_state_client,pose=best_pose)

						open_gripper(self.robot_command_client)

						# Capture and save images to disk
						image_responses = self.image_client.get_image_from_sources(self.sources)

						cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

						cv2.imwrite("./tmp/color_curview_.jpg", cv_visual)

						if self.clip_model == None: #Need to load in CLIP model to run on local image
							self.clip_model, self.clip_preprocess = clip.load(self.config["clip"]["model"])
						priority_queue_vild_dir_cur, priority_queue_clip_dir_cur = get_best_clip_vild_dirs(self.clip_model,self.clip_preprocess,["color_curview_.jpg"],"./tmp",cache_images=False,cache_text=False,cache_path=self.cache_path,img_dir_name="",category_names=[category_name],headless=True)

						#TODO: For now, just get top region and pick
						top_k_item_vild = priority_queue_vild_dir_cur[category_name].get()
						top_k_item_clip = priority_queue_clip_dir_cur[category_name].get()

						ymin, xmin, ymax, xmax = top_k_item_clip[1][3:]

						center_y = int((ymin + ymax)/2.0)
						center_x = int((xmin + xmax)/2.0)

						best_pixel = (center_x, center_y)

						print(best_pixel)
						fig, axs = plt.subplots(1, 2)
						axs[0].imshow(cv_visual)
						axs[1].imshow(top_k_item_clip[1][2])
						plt.savefig('./tmp/crops.png')
						#plt.show()

						input("Execute grasp?")

						arm_object_grasp(self.robot_state_client,self.manipulation_api_client,best_pixel,image_responses[1])

						break #move onto next category



if __name__ == "__main__":
	### Parse arguments from command line
	parser = argparse.ArgumentParser()
	parser.add_argument("-c","--config_path", help="Path to config file", type=str, default="./configs/example.ini")
	args = parser.parse_args()

	nlmap = NLMap(args.config_path)

	### Example things to do 
	#nlmap.viz_pointcloud()
	# nlmap.viz_top_k(viz_2d=False,viz_pointcloud=False)
	# nlmap.go_to_and_pick_top_k("Cup")

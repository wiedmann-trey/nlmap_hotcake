import re
from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    rand_score, 
	adjusted_rand_score,
)

import os
import pandas as pd
import configparser
import csv
from collections import Counter

def cluster_accuracy(labels_gt, labels_pred):
    
		'''
			given a list of expected clusters, and a list of predicted clusters, returns the  accuracy of the predictions 

			inputs: labels_gt, a list of ints representing expected clusters.
					labels_pred, a list of ints representing expected clusters in the same order as labels_gt

			output: a 5-item tuple: (rand index, adjusted rand index, mutual information, normalized mutual information, adjusted mutual information)
			
		'''
		
		# relabeling the images that are labeled by dbscn as noise as being in their own cluster with only one image in each. this makes sure the noise images aren't considered to all be in the same cluster
		adjusted_for_noise = []
		i = 100
		for index in labels_pred:
			if index == -1:
				adjusted_for_noise.append(i)
				i += 1
				continue
			adjusted_for_noise.append(index)
		labels_pred = adjusted_for_noise

		# generating the different accuracy measures		
		RI = rand_score(labels_gt, labels_pred)
		ARI = adjusted_rand_score(labels_gt, labels_pred)

		MI = mutual_info_score(labels_gt, labels_pred)
		NMI = normalized_mutual_info_score(labels_gt, labels_pred)
		AMI = adjusted_mutual_info_score(labels_gt, labels_pred)

		return (RI, ARI, MI, NMI, AMI)

def save_clusters_gts(config, cache_path, embedding_type, eps, samples, image_list):
		'''
			saves a csv with columns: [cluster path, list of all the round truth objects in the cluster]
		'''

		# add the position_x,position_y,position_z to each gt. so maybe do if only_pose save (gt_name, position_x,position_y,position_z)

		print(cache_path, embedding_type, eps, samples)
		df = pd.DataFrame(columns=["batch and cluster number", "number of crops in cluster", "frequency per object in cluster", "ground truth objects"])
		# df = pd.DataFrame()
		embeddings_df = pd.read_csv(f"{cache_path}_embeddings.csv",  dtype={'image_index': 'object'}, index_col=0)

		if config["embedding_type"].getboolean("only_pose"):
				embedding_type = "pose"
		elif config["embedding_type"].getboolean("vild_and_pose"):
			embedding_type = "vild_and_pose"
		elif config["embedding_type"].getboolean("learned"):
			embedding_type = "learned"	
		
		step = config['our_method'].getint('window_step')
		window_size = config['our_method'].getint('window_size')

		############### generating the per batch csv 
		# if config["our_method"].getboolean("KTH_dataset"):
		# myFilePath = f"{cache_path}_per_batch_cluster_{embedding_type}_{samples}_{eps}.csv"
		# print(f'cache path {myFilePath}')
		# # finding the average number of clusters per batch
		# per_batch_df = pd.read_csv(myFilePath)
		# n_rows = len(per_batch_df)
		# print(f'number {n_rows}')
		# print(per_batch_df)
		# n_batches = float(per_batch_df.loc[0, 'Batch number'])
		# print(f'number batches {n_batches}')
		# if n_batches > 0:
		# 	average_cluster_per_batch = (n_rows - n_batches - n_batches) / n_batches
		# else:
		# 	average_cluster_per_batch = n_rows - 2
		# df.loc[len(df.index)] = ["average_number_of_clusters", average_cluster_per_batch, None]

		# get average number of clusters per batch
		cluster_count = 0
		batch_count = 0
		for batch in os.listdir(config["paths"]["cluster_dir"]):
			batch_count += 1
			for cluster in os.listdir(os.path.join(config["paths"]["cluster_dir"], batch)):
				cluster_count += 1
		average_cluster_per_batch = cluster_count/batch_count
		# print(f'len1 {len(df.index)}')
		# df.loc[len(df.index)] = [f"average number of clusters across all batches is {average_cluster_per_batch}", None, None, None]

		# generating the list of ground truth objects and their locations
		for batch in sorted_nicely(os.listdir(config["paths"]["cluster_dir"])):
			for cluster in sorted_nicely(os.listdir(os.path.join(config["paths"]["cluster_dir"], batch))):
				object_gts = []
				if ".html" in cluster:
					continue
				for crop in sorted_nicely(os.listdir((os.path.join(config["paths"]["cluster_dir"], batch, cluster)))):
					# if "combined" not in crop:
					crop_fname = name = f"{config['paths']['cache_dir']}/{crop}" 
					if not config["our_method"].getboolean("KTH_dataset"):
						stripped = crop_fname.strip(".jpeg")+".jpeg"
					subset_df = embeddings_df[embeddings_df['pred_anno_idx'] == f"{config['paths']['cache_dir']}/{crop}"]
					x_df = subset_df['position_x']	
					y_df = subset_df['position_y']	
					z_df = subset_df['position_z']	
					x = x_df.iloc[0]
					y = y_df.iloc[0]
					z = z_df.iloc[0]

					if config["our_method"].getboolean("KTH_dataset"):
						gt_df = subset_df['ground_truth_label_name']	
						gt_label = gt_df.iloc[0]
						object_gts.append((gt_label, x, y, z, subset_df['pred_anno_idx']))
					else:
						object_gts.append((crop, x, y, z))

				object_names = [str(x[0]) for x in object_gts]
				cluster_name = f"{batch}_cluster_{cluster}"

				images_in_cluster = os.listdir(os.path.join(config["paths"]["cluster_dir"], batch, cluster))

				# avoids double counting combined depth/rgb images
				if not config["our_method"].getboolean("KTH_dataset"):
					images_in_cluster = [x for x in images_in_cluster if "color" in x]

				df.loc[len(df.index)] = [cluster_name, len(images_in_cluster), Counter(object_names), object_gts]

		# creating dataframe with list of images in each batch
		batch_df = pd.DataFrame(columns=["batch number", "images in batch"])
		for batch in sorted_nicely(os.listdir(config["paths"]["cluster_dir"])):
			batch_number = int(batch[-1])
			batch_images = image_list[step * batch_number : step * batch_number + window_size]
			batch_df.loc[len(batch_df.index)] = [f"{batch}", batch_images]
	
		# saving the dataframes to csvs
		if config["our_method"].getboolean("KTH_dataset"):
			df.to_csv(f'./cluster_gt_csvs_kosher/cluster_gts_{embedding_type}_eps:{eps}_samples:{samples}_window:{config["our_method"].getint("window_size")}_step:{config["our_method"].getint("window_step")}.csv')
			batch_df.to_csv(f'./cluster_gt_csvs_kosher/batch_images_window:{config["our_method"].getint("window_size")}_step:{config["our_method"].getint("window_step")}_max_images:{config["our_method"].getint("max_images")}.csv')
		else:
			df.to_csv(f'./cluster_csvs_spot/spot_cluster_gts_data:{config["dir_names"]["data"]}_{embedding_type}_eps:{eps}_samples:{samples}_window:{config["our_method"].getint("window_size")}_step:{config["our_method"].getint("window_step")}.csv')
			batch_df.to_csv(f'./cluster_csvs_spot/batch_images:{config["dir_names"]["data"]}_window:{config["our_method"].getint("window_size")}_step:{config["our_method"].getint("window_step")}_max_images:{config["our_method"].getint("max_images")}.csv')
		
		print("end of populating gts")
		exit()

def sorted_nicely(l): 
    """ 
		from https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
		
		Sort the given iterable in the way that humans expect.
	""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# f"{self.cache_path}_per_batch_cluster_{embedding_type}_{samples}_{epsilon}.csv" to get av clusters, just take the length and divide

# config = configparser.ConfigParser()
# config.sections()
# config.read('./configs/example.ini')
# save_clusters_gts(config, './cache/combined_moving_static_KTH', 'pose', 0.25, 5, ['crop1', 'crop2'])
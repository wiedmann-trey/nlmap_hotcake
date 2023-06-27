from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    rand_score, 
	adjusted_rand_score,
)

import os
import pandas as pd

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
		print(f"labels pred: {labels_pred}")

		# generating the different accuracy measures		
		RI = rand_score(labels_gt, labels_pred)
		ARI = adjusted_rand_score(labels_gt, labels_pred)

		MI = mutual_info_score(labels_gt, labels_pred)
		NMI = normalized_mutual_info_score(labels_gt, labels_pred)
		AMI = adjusted_mutual_info_score(labels_gt, labels_pred)

		return (RI, ARI, MI, NMI, AMI)

def save_clusters_gts(config, cache_path, embedding_type, eps, samples):
		'''
			saves a csv with columns: [cluster path, list of all the round truth objects in the cluster]
		'''

		# add the position_x,position_y,position_z to each gt. so maybe do if only_pose save (gt_name, position_x,position_y,position_z)

		df = pd.DataFrame(columns=["cluster path", "gt objects"])
		embeddings_df = pd.read_csv(f"{cache_path}_embeddings.csv",  dtype={'image_index': 'object'}, index_col=0)
		for batch in sorted(os.listdir(config["paths"]["cluster_dir"])):
			for cluster in sorted(os.listdir(os.path.join(config["paths"]["cluster_dir"], batch))):
				object_gts = []
				if ".html" in cluster:
					continue
				print(f'is this path for cluster wrong {str(os.path.join(config["paths"]["cluster_dir"], batch, cluster))}')
				for crop in sorted(os.listdir((os.path.join(config["paths"]["cluster_dir"], batch, cluster)))):
					gt_df = embeddings_df[embeddings_df['pred_anno_idx'] == f"{config['paths']['cache_dir']}/{crop}"]['ground_truth_label_name']	
					gt_label = gt_df.iloc[0]
					object_gts.append(gt_label)
				cluster_name = f"{batch}_cluster_{cluster}"
				df.loc[len(df.index)] = [cluster_name, object_gts]
		
		df.to_csv(f'cluster_gts_{embedding_type}_eps:{eps}_samples:{samples}.csv')
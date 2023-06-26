import cv2
import numpy as np

def intersect_over_union(bb, gt):
    """
    Calculate the fraction of groundtruth bounding box that is
    covered by the bounding box

    We adapt this code from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Parameters
    ----------
    bb : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    gt : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb['x1'] <= bb['x2']
    assert bb['y1'] <= bb['y2']
    assert gt['x1'] <= gt['x2']
    assert gt['y1'] <= gt['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb['x1'], gt['x1'])
    y_top = max(bb['y1'], gt['y1'])
    x_right = min(bb['x2'], gt['x2'])
    y_bottom = min(bb['y2'], gt['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb_area = (bb['x2'] - bb['x1'] + 1) * (bb['y2'] - bb['y1'] + 1)
    gt_area = (gt['x2'] - gt['x1'] + 1) * (gt['y2'] - gt['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    io_gt = intersection_area / float(bb_area + gt_area - intersection_area)
    #print("Bounding box")
    #print(bb)
    #print("Ground truth")
    #print(gt)
    #print(f"Percent overlap {io_gt}")
    assert io_gt >= 0.0
    assert io_gt <= 1.0
    return io_gt

def get_box(path):
    '''
        input: the path to a black and white label image, where the object crop is in white and the rest of the image is black (there can only be one object in the image)
        output: the coordinates of the bounding box of the object in a tuple (xmin, xmax, ymin, ymax)
    '''

    img = cv2.imread(path)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]] # y
    cmin, cmax = np.where(cols)[0][[0, -1]] # x
    return rmin, rmax, cmin, cmax

def extract_pose_from_xml(config, ground_truths, filename, xmin,xmax,ymin,ymax):
		'''
		this method returns the centroid ground truth from the strands dataset
		if there is a corresponding bounding box in the dataset for the one given

		returns:
			groundtruth object name
			groundtruth bounding box (y1,y2,x1,x2)
			groundtruth 3d position (x,y,z,0)
			groundtruth anno index
		'''
		tolerance = float(config['our_method']['bbox_overlap_thresh']) # the tolerated difference between the predicted bounding box and the ground truth (for each corner of the bounding box)

		# you can set the overlap threshhold by changing the "best_overlap" variable
		best_overlap = 0
		best_centroid = None
		if filename in ground_truths:
		# find the corresponding ground truth based on the bounding box
			for box_centroid in ground_truths[filename]:
				indexs = box_centroid[1]
				rmin = indexs[0]
				rmax = indexs[1]
				cmin = indexs[2]
				cmax = indexs[3]
				io_gt = intersect_over_union({'x1': xmin, 'x2': xmax, 'y1': ymin, 'y2': ymax}, {'x1':cmin, 'x2':cmax, 'y1':rmin, 'y2':rmax})
				if io_gt > best_overlap:
					#print(f'io_gt: {io_gt}')
					best_centroid = box_centroid
					best_overlap = io_gt
		else:
			pass
			#print(f"there are no ground truths in the dataset for {filename}")
		if best_centroid and best_overlap > tolerance:
			return best_centroid, best_overlap
		#print(f"there are no corresponding ground truths in the dataset at [xmin,xmax,ymin,ymax = {[xmin,xmax,ymin,ymax]} for {filename}")
		return (None, None, None, None), None
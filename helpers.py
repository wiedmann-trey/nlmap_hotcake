import torch
from torch import nn

def intersect_over_gt(bb, gt):
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
    io_gt = intersection_area / float(gt_area)
    assert io_gt >= 0.0
    assert io_gt <= 1.0
    return io_gt
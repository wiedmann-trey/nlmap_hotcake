'''
reads rgb_0000_label_0.jpg, checks which pixels are white, 
and returns a square bounding box around the white pixels.
'''
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

class Test():
    def __init__(self):
        self.data_dir_path = "/home/ifrah/longterm_semantic_map/combined_moving_static_KTH"
        self.ground_truths = {}
        # dictionary from image name to a list of tuples containing the ground truth label, bounding box coordinates, and centroid of object
        # the the extract pose function uses this dictionary to find a matching image name and a matching bbox for the inputs. If it matches, return the centroid

        for filename in os.listdir(self.data_dir_path):
            end_index = filename.find(".")
            # xml_names = [xml_file for xml_file in os.listdir(self.data_dir_path) \
            # if (filename[0:end_index] in str(xml_file)) and ".xml" in str(xml_file)]

            # populating the ground truth dictionary for each image
            print(filename)
            if ".jpg" in filename and "label" not in filename:
                label_names = [label for label in os.listdir(self.data_dir_path) \
                               if ((filename[0:end_index] in str(label)) and ".jpg" in str(label) and "label" in str(label))]
                i = 0
                for label_name in label_names:
                    rmin, rmax, cmin, cmax = self.get_box(label_name)
                    parsed_xml = ET.parse(os.path.join(self.data_dir_path, label_name[:-3] + "xml"))
                    root = parsed_xml.getroot()
                    # each image is mapped to a tuple of the form:
                    # (string representing ground truth label, list of ints for bounding box coordinates, list of floats for ground truth centroid)
                    # we extract the centroid from the ground truth .xml file
                    if filename not in self.ground_truths:
                        self.ground_truths[filename] = []
                    if "2014" in label_name: 
                        self.ground_truths[filename].append((root.attrib["label"], [rmin, rmax, cmin, cmax], [float(i) for i in root[0].text.split(" ")]))
                    if "2016" in label_name or "2015" in label_name:
                        self.ground_truths[filename].append((root.attrib["label"], [rmin, rmax, cmin, cmax], [float(i) for i in root[2].text.split(" ")]))
                    i += 1
            
        print(self.ground_truths)

    def extract_pose_from_xml(self, filename, xmin,xmax,ymin,ymax):
            '''
            this method returns the centroid ground truth from the strands dataset
            if there is a corresponding bounding box in the dataset for the one given
            '''
            tolerance = 25 # the tolerated difference between the predicted bounding box and the ground truth (for each corner of the bounding box)
            
            # find the corresponding ground truth based on the bounding box
            if filename in self.ground_truths:
                for box_centroid in self.ground_truths[filename]:
                    indexs = box_centroid[1]
                    rmin = indexs[0]
                    rmax = indexs[1]
                    cmin = indexs[2]
                    cmax = indexs[3]

                    if abs(cmin - ymin) < tolerance and abs(cmax - ymax) < tolerance \
                        and abs(rmax - xmax) < tolerance and abs(rmin - xmin) < tolerance:
                        return box_centroid[2]
                print(f"there are no corresponding ground truths in the dataset for {filename}")
            else:
                    print(f"there are no ground truths in the dataset for {filename}")

    def get_box(self, filename):
        path = os.path.join("/home/ifrah/longterm_semantic_map/combined_moving_static_KTH", filename)
        # path = filename
        img = cv2.imread(path)
        # print(img.shape)
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        # print(rows, "space", cols)
        # print(np.any(rows), np.any(cols))
        rmin, rmax = np.where(rows)[0][[0, -1]] # y
        cmin, cmax = np.where(cols)[0][[0, -1]] # x

        # return rmin, rmax, cmin, cmax
        return [cmin, rmin, cmax, rmax]
    
    def IOU(self, box1, box2):
        """ We assume that the box follows the format:
            box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
            where (x1,y1) and (x3,y3) represent the top left coordinate,
            and (x2,y2) and (x4,y4) represent the bottom right coordinate """
        x1, y1, x2, y2 = box1	
        x3, y3, x4, y4 = box2
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou

test = Test()
# assert test.extract_pose_from_xml("20140905_patrol_run_30_room_7_rgb_0001.jpg", 304, 367, 328, 407) == [0.633533, 0.410255, 2.12958, 0.0]
# assert test.get_box("20140905_patrol_run_30_room_7_rgb_0001_label_0.jpg") == (304, 367, 328, 407)

# # test for picture with multiple labels
# assert test.extract_pose_from_xml("20140910_patrol_run_79_room_2_rgb_0006.jpg", 176, 271, 264, 343) == [1.45471, 0.762191, -0.406816, 0.0]
# assert test.extract_pose_from_xml("20140910_patrol_run_79_room_2_rgb_0006.jpg", 160, 239, 312, 407) == [2.87556, 1.47612, -1.22137, 0.0]

# assert test.extract_pose_from_xml("20140908_patrol_run_41_room_3_rgb_0009.jpg", 208, 247, 368, 463) == [-0.418084, 1.9244, -2.21457, 0.0]
# assert test.extract_pose_from_xml("20140908_patrol_run_41_room_3_rgb_0009.jpg", 128, 463, 296, 543) == [-0.193949, 1.50212, -1.25298, 0.0]
# assert test.extract_pose_from_xml("20140908_patrol_run_41_room_3_rgb_0009.jpg", 232, 271, 416, 447) == [-0.394846, 1.80753, -1.88986, 0.0]


# print(test.get_box("20140911_patrol_run_84_room_3_rgb_0010_label_0.jpg"))
# box1 = test.get_box("20140911_patrol_run_84_room_3_rgb_0010_label_0.jpg")
# box2 = test.get_box("20140911_patrol_run_84_room_3_rgb_0010_label_1.jpg")
# print(f"box1: {box1}, box2: {box2}")
# x1, y1, x2, y2 = box1	
# x3, y3, x4, y4 = box2
# print(test.IOU(box1, box2))
# print(test.IOU([0, 4, 5, 6], [100, 200, 300, 400]))
print(test.get_box("20140820_patrol_run_2_room_1_rgb_0009_label_1.jpg"))

from helpers import intersect_over_gt

print(intersect_over_gt({'x1': 463, 'x2': 486, 'y1': 249, 'y2': 276}, {'x1':456, 'x2':487, 'y1':240, 'y2':279}))
# [463, 486, 249, 276]
# print(test.get_box("20140911_patrol_run_84_room_3_rgb_0012_label_0.jpg")) # high x low y

# print(test.get_box("/home/ifrah/Desktop/contract-fr-superb-101-sq.jpg"))
# print(test.get_box("20140820_patrol_run_2_room_1_rgb_0007_label_0.jpg"))


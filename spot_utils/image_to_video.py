import cv2 
import os 

def images_to_video(image_folder, output_filename, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key = lambda x: str(x.split(".")[0]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()

# Path to image folder
# [TODO] Change this to make absolute path
image_folder = '2023-06-27-scan-1'
output_filename = image_folder + '_video.mp4'

# Set the Frame Rate
fps = 12
images_to_video(image_folder, output_filename, fps)
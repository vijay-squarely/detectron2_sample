import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Set up configuration and model
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.WEIGHTS = "http://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.DEVICE = "cuda"  # Use GPU if available

# Create a predictor using the model
predictor = DefaultPredictor(cfg)

# Load an image from the input folder
img = cv2.imread("input/image.jpeg")

# Make predictions
outputs = predictor(img)

# Visualize the predictions
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save the result
output_image = "output.jpeg"
cv2.imwrite(output_image, v.get_image()[:, :, ::-1])

print(f"Object detection complete. Output saved to {output_image}")
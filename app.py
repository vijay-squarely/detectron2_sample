import torch
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import pairwise_ioa

# Define custom class mappings
damage_class_map = {0: 'damage'}
parts_class_map = {0: 'headlamp', 1: 'rear_bumper', 2: 'door', 3: 'hood', 4: 'front_bumper'}

# Register custom metadata for damage detection
damage_metadata = MetadataCatalog.get("damage_dataset")
damage_metadata.thing_classes = list(damage_class_map.values())

# Register custom metadata for parts detection
parts_metadata = MetadataCatalog.get("parts_dataset")
parts_metadata.thing_classes = list(parts_class_map.values())

def estimate_cost(damage_instances, parts_instances, parts_metadata):
    damage_cost = 250  # Base cost for damage
    labor_charge = 250  # Base labor charge for any detection
    parts_cost = {
        "headlamp": 1500,
        "rear_bumper": 2000,
        "door": 3000,
        "hood": 2500,
        "front_bumper": 1800
    }

    total_cost = 0
    labor_needed = False
    detected_parts = []
    damage_in_parts = [False] * len(damage_instances) 

    # Calculate cost based on detected parts
    if len(parts_instances) > 0:
        for i in range(len(parts_instances)):
            part_class_id = parts_instances.pred_classes[i].item()
            part_name = parts_metadata.thing_classes[part_class_id]
            detected_parts.append(part_name)
            estimated_part_cost = parts_cost.get(part_name, 0)
            total_cost += estimated_part_cost

            # Set labor flag if any detection occurs
            labor_needed = True

            # Display part detection messages
            st.write(f"Part detected: {part_name}, Estimated part repair cost: ${estimated_part_cost}")

            for j in range(len(damage_instances)):
                damage_box = damage_instances.pred_boxes[j].tensor.numpy()
                part_box = parts_instances.pred_boxes[i].tensor.numpy()

                # Check for intersection
                if (
                    damage_box[0][0] < part_box[0][2] and
                    damage_box[0][2] > part_box[0][0] and
                    damage_box[0][1] < part_box[0][3] and
                    damage_box[0][3] > part_box[0][1]
                ):
                    damage_in_parts[j] = True 

    if len(damage_instances) > 0:
        for i in range(len(damage_instances)):
            if not damage_in_parts[i]:
                total_cost += damage_cost
                st.write(f"Damage detected, Estimated damage cost: ${damage_cost}")

                labor_needed = True

    if labor_needed:
        total_cost += labor_charge
        st.write(f"Labor charge applied: ${labor_charge}")

    if total_cost == 0:
        return "No damage or parts detected, no cost estimation available."
    else:
        return f"Total estimated repair cost: ${total_cost}"

# Function to predict both damage and parts
def predict(image_np):
    # Damage Model Configuration
    cfg_damage = get_cfg()
    cfg_damage.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_damage.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_damage.MODEL.WEIGHTS = "/home/niyas/automated car damage detection/models/damage_org.pth"
    cfg_damage.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg_damage.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Damage predictor
    damage_predictor = DefaultPredictor(cfg_damage)
    damage_outputs = damage_predictor(image_np)
    damage_instances = damage_outputs["instances"].to("cpu")
    high_conf_damage = damage_instances[damage_instances.scores > 0.5]

    # Parts Model Configuration
    cfg_parts = get_cfg()
    cfg_parts.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_parts.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg_parts.MODEL.WEIGHTS = "/home/niyas/automated car damage detection/models/parts_org.pth"
    cfg_parts.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg_parts.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parts predictor
    parts_predictor = DefaultPredictor(cfg_parts)
    parts_outputs = parts_predictor(image_np)
    parts_instances = parts_outputs["instances"].to("cpu")
    high_conf_parts = parts_instances[parts_instances.scores > 0.75]

    return high_conf_damage, high_conf_parts

# Streamlit UI
st.title("Automated Car Damage Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run predictions
    high_conf_damage, high_conf_parts = predict(image_np)

    # IoU filtering logic (shared for both buttons)
    if len(high_conf_damage) > 0:
        damage_boxes = high_conf_damage.pred_boxes
        parts_boxes = high_conf_parts.pred_boxes

        iou_matrix = pairwise_ioa(parts_boxes, damage_boxes)

        filtered_indices = []
        for i in range(iou_matrix.shape[0]):
            if iou_matrix[i].max() > 0:
                filtered_indices.append(i)

        filtered_parts = high_conf_parts[filtered_indices]
    else:
        filtered_parts = []

    # "Predict" button logic
    if st.button('Predict'):
        if len(high_conf_damage) > 0:
            # Visualize and display the results
            v_damage = Visualizer(image_np[:, :, ::-1], metadata=damage_metadata, instance_mode=ColorMode.IMAGE)
            v_damage = v_damage.draw_instance_predictions(high_conf_damage)
            damage_result = v_damage.get_image()[:, :, ::-1]

            v_parts = Visualizer(image_np[:, :, ::-1], metadata=parts_metadata, instance_mode=ColorMode.IMAGE)
            v_parts = v_parts.draw_instance_predictions(filtered_parts)
            parts_result = v_parts.get_image()[:, :, ::-1]

            combined_result = cv2.addWeighted(damage_result, 0.5, parts_result, 0.5, 0)
            st.image(combined_result, caption='Damage and Parts Detection with Masks', use_column_width=True)

            # Display cost estimation
            estimated_cost = estimate_cost(high_conf_damage, filtered_parts, parts_metadata)
            st.write(estimated_cost)
        else:
            st.write("No damage detected, skipping parts prediction.")

    # "Show" button logic (with IoU filtering)
    if st.button('Predict sepretely'):
        if len(high_conf_damage) > 0:
            # Visualize and display filtered damage and parts
            v_damage_raw = Visualizer(image_np[:, :, ::-1], metadata=damage_metadata, instance_mode=ColorMode.IMAGE)
            v_damage_raw = v_damage_raw.draw_instance_predictions(high_conf_damage)
            raw_damage_result = v_damage_raw.get_image()[:, :, ::-1]
            st.image(raw_damage_result, caption='Damage Detection Only', use_column_width=True)

            v_parts_raw = Visualizer(image_np[:, :, ::-1], metadata=parts_metadata, instance_mode=ColorMode.IMAGE)
            v_parts_raw = v_parts_raw.draw_instance_predictions(filtered_parts)
            raw_parts_result = v_parts_raw.get_image()[:, :, ::-1]
            st.image(raw_parts_result, caption='Parts Detection Only (Filtered)', use_column_width=True)
        else:
            st.write("No damage detected, skipping parts prediction.")
 
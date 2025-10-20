import os
import json
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import argparse


# === CONFIGURATION ===
parser = argparse.ArgumentParser(description="Convert BridgeData to HDF5 format.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input root directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output root directory")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

# Paths
json_path = os.path.join(input_dir, "labels/train.json")                                    # JSON file with annotations
video_dir = os.path.join(input_dir, "20bn-something-something-v2")                          # Directory containing .webm files
os.makedirs(output_dir, exist_ok=True)

# Load JSON annotations
with open(json_path, "r") as f:
    annotations = json.load(f)

# Define function to extract frames from a video
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)

# Loop through annotations
for entry in tqdm(annotations):
    video_id = entry["id"]
    label = entry["label"]
    video_file = os.path.join(video_dir, f"{video_id}.webm")
    
    if not os.path.exists(video_file):
        print(f"Video not found: {video_file}")
        continue

    images = read_video_frames(video_file)
    T = len(images)
    if T == 0:
        print(f"No frames in {video_file}")
        continue

    # Prepare dummy arrays
    shape_image = (T, 128, 128, 3)
    shape_depth = (T, 128, 128, 1)
    shape_3d = (T, 4, 4)
    shape_intrinsic = (T, 3, 3)
    shape_gripper = (T, 2)
    shape_pos = (T, 3)
    shape_1d = (T,)
    
    depth_dummy = np.zeros(shape_depth, dtype=np.uint16)
    extrinsic_dummy = np.zeros(shape_3d, dtype=np.float64)
    intrinsic_dummy = np.zeros(shape_intrinsic, dtype=np.float64)
    hand_mat_dummy = np.zeros(shape_3d, dtype=np.float32)
    hand_mat_inv_dummy = np.zeros(shape_3d, dtype=np.float32)
    gripper_qpos_dummy = np.zeros(shape_gripper, dtype=np.float64)
    eef_pos_dummy = np.zeros(shape_pos, dtype=np.float64)

    reward_dummy = np.zeros(shape_1d, dtype=np.float64)
    success_dummy = np.zeros(shape_1d, dtype=bool)
    terminated_dummy = np.zeros(shape_1d, dtype=bool)
    truncated_dummy = np.zeros(shape_1d, dtype=bool)
    states_dummy = np.array([], dtype=np.float64)

    actions_dummy = np.zeros((T, 7), dtype=np.float64)
    abs_actions_dummy = np.zeros((T, 7), dtype=np.float64)

    
    # Output HDF5 path
    file_name = label.replace('/', '').replace(' ', '_')
    h5_file = os.path.join(output_dir, f"{str(video_id)}_{file_name}.hdf5")
     
    with h5py.File(h5_file, 'w') as f:
        grp = f.create_group("data/demo_0")
        obs = grp.create_group("obs")

        obs.create_dataset("agentview_image", data=images, dtype=np.uint8)
        obs.create_dataset("robot0_eye_in_hand_image", data=images, dtype=np.uint8)

        obs.create_dataset("agentview_depth", data=depth_dummy)
        obs.create_dataset("robot0_eye_in_hand_depth", data=depth_dummy)

        obs.create_dataset("agentview_extrinsic", data=extrinsic_dummy)
        obs.create_dataset("robot0_eye_in_hand_extrinsic", data=extrinsic_dummy)

        obs.create_dataset("agentview_intrinsic", data=intrinsic_dummy)
        obs.create_dataset("robot0_eye_in_hand_intrinsic", data=intrinsic_dummy)

        obs.create_dataset("hand_mat", data=hand_mat_dummy)
        obs.create_dataset("hand_mat_inv", data=hand_mat_inv_dummy)

        obs.create_dataset("robot0_gripper_qpos", data=gripper_qpos_dummy)
        obs.create_dataset("robot0_eef_pos", data=eef_pos_dummy)

        grp.create_dataset("abs_actions", data=abs_actions_dummy)
        grp.create_dataset("actions", data=actions_dummy)
        grp.create_dataset("reward", data=reward_dummy)
        grp.create_dataset("success", data=success_dummy)
        grp.create_dataset("terminated", data=terminated_dummy)
        grp.create_dataset("truncated", data=truncated_dummy)
        grp.create_dataset("states", data=states_dummy)



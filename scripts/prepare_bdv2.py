import os
import pickle
import json
import h5py
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import argparse
import re

# === CONFIGURATION ===
parser = argparse.ArgumentParser(description="Convert BridgeData to HDF5 format.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input root directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output root directory")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# === HELPERS ===
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None

def clean_instruction(lang_path):
    if not os.path.exists(lang_path):
        return "unnamed_task"
    with open(lang_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if not line.lower().startswith("confidence:"):
            return line.strip()
    return "unnamed_task"

def safe_filename(text, idx):
    cleaned = text.rstrip('.').replace("/", "_").replace(" ", "_")
    return f"{idx}_{cleaned}"
    # return re.sub(r"[^a-zA-Z0-9_]+", "_", text.replace(" ", "_")).strip("_")

def convert_traj_to_hdf5(traj_path, save_path):
    policy_out = load_pickle(os.path.join(traj_path, "policy_out.pkl"))
    obs_dict = load_pickle(os.path.join(traj_path, "obs_dict.pkl"))
    lang_path = os.path.join(traj_path, "lang.txt")
    images_dir = os.path.join(traj_path, "images0")

    if policy_out is None or obs_dict is None:
        print(f"[SKIP] Missing critical files in {traj_path}")
        return

    instruction_text = clean_instruction(lang_path)
    T = len(policy_out)

    # Load images
    image_paths = sorted(glob(os.path.join(images_dir, "*.jpg")))
    if len(image_paths) >= T:
        images = np.stack([np.array(Image.open(p).resize((128, 128))) for p in image_paths[:T]], axis=0)
    else:
        images = np.zeros((T, 128, 128, 3), dtype=np.uint8)

    # Dummy arrays
    shape_depth = (T, 128, 128, 1)
    shape_3d = (T, 4, 4)
    shape_intrinsic = (T, 3, 3)
    shape_gripper = (T, 2)
    shape_pos = (T, 3)
    shape_1d = (T,)
    shape_action = (T, 7)

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
    states_dummy = np.zeros((T,), dtype=np.float64)

    # Actions
    actions = np.array([d["actions"] for d in policy_out], dtype=np.float64)
    abs_actions = actions.copy()

    # Save
    with h5py.File(save_path, 'w') as f:
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

        grp.create_dataset("actions", data=actions)
        grp.create_dataset("abs_actions", data=abs_actions)
        grp.create_dataset("reward", data=reward_dummy)
        grp.create_dataset("success", data=success_dummy)
        grp.create_dataset("terminated", data=terminated_dummy)
        grp.create_dataset("truncated", data=truncated_dummy)
        grp.create_dataset("states", data=states_dummy)

        f["data"].attrs["language_instruction"] = instruction_text
        f["data/demo_0"].attrs["num_samples"] = T

    print(f"✅ Converted and saved: {save_path}")

# === RECURSIVE SCAN ===
input_cols = os.listdir(input_dir)
idx = 1
for input_col in tqdm(input_cols):
    for root, dirs, files in tqdm(os.walk(os.path.join(input_dir, input_col))):
        if all(os.path.exists(os.path.join(root, name)) for name in ["policy_out.pkl", "obs_dict.pkl", "lang.txt"]):
            instruction = clean_instruction(os.path.join(root, "lang.txt"))
            filename = safe_filename(instruction, idx) + ".hdf5"
            out_path = os.path.join(output_dir, filename)
            convert_traj_to_hdf5(root, out_path)
            idx += 1
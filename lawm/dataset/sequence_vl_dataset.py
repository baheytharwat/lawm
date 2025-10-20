import torch
import random
from torch.utils.data import Dataset
from typing import Optional, Union, Dict, Any, List
import h5py


class SequenceVLDataset(Dataset):
    """Dataset wrapper that adds vision-language task information to sequence data.
    
    Args:
        sequence_dataset: Base sequence dataset
        task_emb: Optional task embedding tensor or list of tensors
        lang_inst: Optional language instruction string or list of strings
        task_id: Optional task ID (int/tensor) or list of task IDs
    """
    def __init__(
        self, 
        sequence_dataset: Dataset,
        task_emb: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        lang_inst: Optional[Union[str, List[str]]] = None,
        task_id: Optional[Union[int, torch.Tensor, List[Union[int, torch.Tensor]]]] = None
    ):
        self.sequence_dataset = sequence_dataset
        # Convert all inputs to lists
        self.task_emb = [task_emb] if task_emb is not None and not isinstance(task_emb, list) else task_emb
        self.lang_inst = [lang_inst] if lang_inst is not None and not isinstance(lang_inst, list) else lang_inst
        self.task_id = [task_id] if task_id is not None and not isinstance(task_id, list) else task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return_dict = self.sequence_dataset.__getitem__(idx)
        
        # Add any provided task information to the return dict, sampling from lists
        if self.task_emb is not None:
            return_dict["task_emb"] = random.choice(self.task_emb)
        if self.lang_inst is not None:
            return_dict["lang_inst"] = random.choice(self.lang_inst)
        if self.task_id is not None:
            return_dict["task_id"] = random.choice(self.task_id)
            
        return return_dict

class CustomVLADataset(Dataset):
    def __init__(self, 
                 h5_path: str, 
                 task_id: int, 
                 task_emb: torch.Tensor, 
                 chunk_size: int = 1, 
                 apply_wm: bool = False):
        super().__init__()
        self.h5_path = h5_path
        self.task_id = task_id
        self.task_emb = task_emb
        self.chunk_size = chunk_size
        self.apply_wm = apply_wm
        self.frame_stack = 1
        self.frame_stack_keys = ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_gripper_qpos']

        with h5py.File(h5_path, 'r') as f:
            self.actions = f['data/demo_0/actions'][:]
            self.obs = {
                k: f[f'data/demo_0/obs/{k}'][:] for k in [
                    'agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
                    'robot0_gripper_qpos', 'hand_mat', 'hand_mat_inv',
                    'agentview_extrinsic', 'robot0_eye_in_hand_extrinsic',
                    'agentview_intrinsic', 'robot0_eye_in_hand_intrinsic'
                ]
            }

        self.length = len(self.actions) - 1 # - self.chunk_size
        self.n_demos = 1
        self.total_num_sequences = self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actions = torch.tensor(self.actions[idx+1:idx+self.chunk_size+1], dtype=torch.float32)
        if len(actions) < self.chunk_size:
            last_action_padded = actions[-1:].repeat(self.chunk_size - len(actions), 1)
            actions = torch.cat([actions, last_action_padded], dim=0)

        obs = {}
        for k, v in self.obs.items():
            if k in self.frame_stack_keys:
                data = v[idx:idx+self.frame_stack]
                if data.ndim == 4:  # image: (T, H, W, C)
                    tensor = torch.tensor(data, dtype=torch.uint8).permute(0, 3, 1, 2)  # (T, C, H, W)
                else:  # low-dim: (T, D)
                    tensor = torch.tensor(data, dtype=torch.float32)  # (T, D)
            else:
                tensor = torch.tensor(v[idx], dtype=torch.float32)
            obs[k] = tensor

        data_dict = {
            'actions': actions,
            'obs': obs,
            'task_emb': self.task_emb.clone(),
            'task_id': self.task_id
        }

        if self.apply_wm:
            wm_obs = self.obs["agentview_image"][idx+1:idx+self.chunk_size+1]
            wm_obs = torch.tensor(wm_obs, dtype=torch.uint8).permute(0, 3, 1, 2)  # (T, C, H, W)
            if len(wm_obs) < self.chunk_size:
                last_obs_padded = wm_obs[-1:].repeat(self.chunk_size - len(wm_obs), 1, 1, 1) ## 1s are number of repeates along each dim
                wm_obs = torch.cat([wm_obs, last_obs_padded], dim=0)

            is_first = torch.zeros(len(wm_obs), dtype=bool)
            is_first[0] = True

            data_dict["wm_obs"] = wm_obs
            data_dict["is_first"] = is_first

        return data_dict
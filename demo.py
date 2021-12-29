'''
    PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging [IJCAI-21]
    https://reyuwei.github.io/proj/piano
'''

import torch
import trimesh
import numpy as np
from PIANOLayer import PIANOLayer

if __name__ == "__main__":
    device = torch.zeros(1).device
    pianolayer = PIANOLayer(r"piano_model/PIANO_RIGHT_dict.pkl", device, shape_ncomp=2)
    zero_pose = pianolayer.zero_pose
    zero_beta = pianolayer.zero_beta

    demo_pose = np.load(r"piano_model/demo_pose.pkl", allow_pickle=True)
    demo_pose = torch.from_numpy(demo_pose).float()
    batch_size = demo_pose.shape[0]
    random_beta = (torch.rand(batch_size, pianolayer.shape_ncomp)-0.5) * 100
    print(random_beta)

    v, j = pianolayer.forward(demo_pose, random_beta)
    bone_mesh = trimesh.Trimesh(v[0], pianolayer.faces, process=False)
    bone_mesh.export("demo_piano.obj")


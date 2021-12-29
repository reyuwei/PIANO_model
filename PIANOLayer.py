import torch
import numpy as np
from utils import th_posemap_axisang_2output, batch_rodrigues, th_with_zeros, th_pack

class PIANOLayer(torch.nn.Module):
    def __init__(self, piano_dict_path, device, shape_ncomp=30, root_joint_idx=0):
        super(PIANOLayer, self).__init__()
        self.device = device
        self.identity_rot = torch.eye(3).to(self.device)

        piano_dict = np.load(piano_dict_path, allow_pickle=True)

        self.shape_ncomp = shape_ncomp
        self.ROOT_JOINT_IDX = root_joint_idx
        self.STATIC_JOINT_NUM = piano_dict['STATIC_JOINT_NUM']
        self.STATIC_BONE_NUM = piano_dict['STATIC_BONE_NUM']
        self.JOINT_ID_BONE_DICT = piano_dict['JOINT_ID_BONE_DICT']

        self.mean_v = torch.from_numpy(piano_dict['shape_mean']).float().to(self.device)
        self.faces =  torch.from_numpy(piano_dict['faces']).float().to(self.device)
        self.jreg =  torch.from_numpy(piano_dict['jreg']).float().to(self.device)
        self.shape_basis =  torch.from_numpy(piano_dict['shape_basis']).float().to(self.device)
        self.skinning_weights = torch.from_numpy(piano_dict['sw']).float().to(self.device)
        
        self.zero_pose = torch.zeros([1, self.STATIC_JOINT_NUM, 3]).to(self.device)
        self.zero_beta = torch.zeros(1, self.shape_ncomp).to(self.device)


        # Kinematic chain params
        kinetree = piano_dict['JOINT_PARENT_ID_DICT']
        self.kintree_parents = []
        for i in range(self.STATIC_JOINT_NUM):
            self.kintree_parents.append(kinetree[i])

    def compute_warp(self, batch_size, points, full_trans_mat):
        if points.shape[0] != batch_size:
            points = points.repeat(batch_size, 1, 1)
        if self.skinning_weights.shape[0] != batch_size:
            skinning_weights = self.skinning_weights.repeat(batch_size, 1, 1)

        th_T = torch.einsum('bijk,bkt->bijt',full_trans_mat, skinning_weights.permute(0, 2, 1))
        th_rest_shape_h = torch.cat([points.transpose(2, 1),
                                     torch.ones((batch_size, 1, points.shape[1]), dtype=skinning_weights.dtype,
                                                device=skinning_weights.device), ], 1)
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        return th_verts

    def forward(self, pose, beta, root_trans=None, global_scale=None):
        """
        Takes points in R^3 and first applies relevant pose and shape blend shapes.
        Then performs skinning.
        """
        batch_size = pose.shape[0]

        th_v_shaped = (self.shape_basis[:self.shape_ncomp].T @ beta.T).view(-1, 3, batch_size).permute(2, 0, 1) + self.mean_v.unsqueeze(0).repeat(batch_size, 1, 1)
        th_j = torch.matmul(self.jreg, th_v_shaped)

        # Convert axis-angle representation to rotation matrix rep.
        _, th_rot_map = th_posemap_axisang_2output(pose.view(batch_size, -1))
        th_full_pose = pose.view(batch_size, -1, 3)
        root_rot = batch_rodrigues(th_full_pose[:, 0]).view(batch_size, 3, 3)

        # Global rigid transformation
        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(self.STATIC_JOINT_NUM - 1):
            i_val_joint = int(i + 1)
            if i_val_joint in self.JOINT_ID_BONE_DICT:
                i_val_bone = self.JOINT_ID_BONE_DICT[i_val_joint]
                joint_rot = th_rot_map[:, (i_val_bone - 1) * 9:i_val_bone * 9].contiguous().view(batch_size, 3, 3)
            else:
                joint_rot = self.identity_rot.repeat(batch_size, 1, 1)

            joint_j = th_j[:, i_val_joint, :].contiguous().view(batch_size, 3, 1)
            parent = self.kintree_parents[i_val_joint]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))

        th_results_global = th_results
        th_results2 = torch.zeros((batch_size, 4, 4, self.STATIC_JOINT_NUM), dtype=root_j.dtype, device=root_j.device)

        for i in range(self.STATIC_JOINT_NUM):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat([th_j[:, i], padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)
        
        th_verts = self.compute_warp(batch_size, th_v_shaped, th_results2)
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # global scaling
        if global_scale is not None:
            center_joint = th_jtr[:, self.ROOT_JOINT_IDX].unsqueeze(1)
            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint

            verts_scale = global_scale.expand(th_verts.shape[0], th_verts.shape[1])
            verts_scale = verts_scale.unsqueeze(2).repeat(1, 1, 3)
            th_verts = th_verts * verts_scale
            th_verts = th_verts + center_joint

            j_scale = global_scale.expand(th_jtr.shape[0], th_jtr.shape[1])
            j_scale = j_scale.unsqueeze(2).repeat(1, 1, 3)
            th_jtr = th_jtr * j_scale
            th_jtr = th_jtr + center_joint

        # global translation
        if root_trans is not None:
            root_position = root_trans.view(batch_size, 1, 3)
            center_joint = th_jtr[:, self.ROOT_JOINT_IDX].unsqueeze(1)
            offset = root_position - center_joint
            
            th_jtr = th_jtr + offset
            th_verts = th_verts + offset

        return th_verts, th_jtr


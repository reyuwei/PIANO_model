import torch

def quat2aa(quats):
    """
    Convert wxyz quaternions to angle-axis representation
    :param quats:
    :return:
    """
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = torch.atan2(_sin, _cos) * 2
    return axis * angle.unsqueeze(-1)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_aa2quat(axisang):
    # w, x, y, z
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return quat

def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def th_posemap_axisang_2output(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb - 1):
        joint_idx_val = joint_idx + 1
        axis_ang = pose_vectors[:, joint_idx_val * 3:(joint_idx_val + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    # rot_mats = torch.stack(rot_mats, 1).view(-1, 15 *9)
    rot_mats = torch.cat(rot_mats, 1)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats

def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
        rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results

def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res

def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res

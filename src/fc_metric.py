import numpy as np

def np_normalize_vector(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def np_normal_to_rot(
    axis_0, rot_base1=np.array([[0, 1, 0]]), rot_base2=np.array([[0, 0, 1]])
):
    proj_xy = np.abs(np.sum(axis_0 * rot_base1, axis=-1, keepdims=True))
    axis_1 = np.where(proj_xy > 0.99, rot_base2, rot_base1)

    axis_1 = np_normalize_vector(
        axis_1 - np.sum(axis_1 * axis_0, axis=-1, keepdims=True) * axis_0
    )
    axis_2 = np.cross(axis_0, axis_1, axis=-1)

    return np.stack([axis_0, axis_1, axis_2], axis=-1)


def build_grasp_matrix(pos, normal, origin=np.array([0, 0, 0])):
    rot = np_normal_to_rot(normal)
    axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]

    # Normalize contact position
    relative_pos = pos - origin

    grasp_matrix = np.zeros((pos.shape[0], 6, 3))
    grasp_matrix[:, :3, 0] = axis_0
    grasp_matrix[:, :3, 1] = axis_1
    grasp_matrix[:, :3, 2] = axis_2
    grasp_matrix[:, 3:, 0] = np.cross(relative_pos, axis_0, axis=-1)
    grasp_matrix[:, 3:, 1] = np.cross(relative_pos, axis_1, axis=-1)
    grasp_matrix[:, 3:, 2] = np.cross(relative_pos, axis_2, axis=-1)
    return grasp_matrix

def calcu_dfc_metric(contact_pos, contact_normal, enable_density=True):
    grasp_matrix = build_grasp_matrix(contact_pos, contact_normal)
    if enable_density:
        cos_theta = (contact_normal[:, None, :] * contact_normal[:, :, None]).sum(
            axis=-1
        )
        density = (
            1
            / np.clip(
                np.clip(cos_theta, a_min=0, a_max=100).sum(axis=-1),
                a_min=1e-4,
                a_max=100,
            )[:, None]
        )
        dfc_metric = np.linalg.norm(np.sum(grasp_matrix[:, :, 0] * density, axis=0))
    else:
        dfc_metric = np.linalg.norm(np.sum(grasp_matrix[:, :, 0], axis=0))
    return dfc_metric

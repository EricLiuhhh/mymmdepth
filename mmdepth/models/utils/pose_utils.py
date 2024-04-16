import torch

'''
Pose regression layer
'''
def pose_matrix(v, rotation_parameterization='axis', invert=False):
    '''
    Convert 6 DoF parameters to transformation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 6 vector in the order of tx, ty, tz, rx, ry, rz
        rotation_parameterization : str
            axis
    Returns:
        torch.Tensor[float32] : N x 4 x 4 homogeneous transformation matrix
    '''

    # Select N x 3 element rotation vector
    r = v[..., :3]
    # Select N x 3 element translation vector
    t = v[..., 3:]

    if rotation_parameterization == 'axis':
        Rt = transformation_from_parameters(torch.unsqueeze(r, dim=1), t, invert)
    else:
        raise ValueError('Unsupported rotation parameterization: {}'.format(rotation_parameterization))

    return Rt


'''
Utility functions for rotation
'''
def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def homogeneous_transformation(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    # return transform.dot(points.T).T
    return (transform @ points.T).T

def project_3d_to_2d(points: torch.Tensor, projection_matrix: torch.Tensor):
    """
This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    #uvw = projection_matrix.dot(points.T)
    # uvw /= uvw[2]
    uvw = projection_matrix @ points.T
    uvw = uvw / uvw[2, :]
    uvs = uvw[:2].T
    uvs = torch.round(uvs).to(int)

    return uvs


def canvas_crop(points, image_size, points_depth=None):
    """
This function filters points that lie outside a given frame size.
    :param points: Input points to be filtered.
    :param image_size: Size of the frame.
    :param points_depth: Filters also depths smaller than 0.
    :return: Filtered points.
    """
    idx = points[:, 0] > 0
    idx = torch.logical_and(idx, points[:, 0] < image_size[1])
    idx = torch.logical_and(idx, points[:, 1] > 0)
    idx = torch.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = torch.logical_and(idx, points_depth > 0)

    return idx

def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape, filter_points=True):
    """
A helper function which projects a point clouds specific to the dataset to the camera image frame.
    :param point_cloud: Point cloud to be projected.
    :param t_camera_pcl: Transformation from the pcl frame to the camera frame.
    :param camera_projection_matrix: The 4x4 camera projection matrix.
    :param image_shape: Size of the camera image.
    :return: Projected points, and the depth of each point.
    """
    point_homo = torch.hstack((point_cloud[:, :3],
                            torch.ones((point_cloud.shape[0], 1),
                                    dtype=torch.float32, device=point_cloud.device)))

    radar_points_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=t_camera_pcl)

    point_depth = radar_points_camera_frame[:, 2]

    if camera_projection_matrix.shape == (3, 3):
        temp = camera_projection_matrix.new_zeros((4, 4))
        temp[:3, :3] = camera_projection_matrix
        temp[-1, -1] = 1
        camera_projection_matrix = temp

    uvs = project_3d_to_2d(points=radar_points_camera_frame,
                           projection_matrix=camera_projection_matrix)

    if filter_points:
        filtered_idx = canvas_crop(points=uvs,
                                image_size=image_shape,
                                points_depth=point_depth)
        uvs = uvs[filtered_idx]
        point_depth = point_depth[filtered_idx]
        return uvs, point_depth, filtered_idx
    else:
        return uvs, point_depth

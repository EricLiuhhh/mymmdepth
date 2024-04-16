from typing import List
import numpy as np
import torch
from mmengine.model import BaseModule
from mmdepth.registry import MODELS

from gaussian_splatting.scene import GaussianModel, dataset_readers, cameras
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import focal2fov, BasicPointCloud
# from mmdepth.third_party.gaussian_splatting.scene import GaussianModel, dataset_readers, cameras
# from mmdepth.third_party.gaussian_splatting.render import render
# from mmdepth.third_party.gaussian_splatting.utils.graphics_utils import focal2fov, BasicPointCloud
from ..utils.pose_utils import project_pcl_to_image

class PipelineParams():
    def __init__(self, convert_SHs_python=False, compute_cov3D_python=False, debug=False):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


@MODELS.register_module()
class GaussianPrefiller(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.gaussian = GaussianModel(3)

    def forward(self, inputs, data_samples):
        gaussian_params = inputs['gaussian_params']
        xyz, rot, scale, opa = gaussian_params['xyz'], gaussian_params['rot'], gaussian_params['sacle'], gaussian_params['opa']
        coords = inputs['voxels']['coords']
        bs = len(data_samples)
        for b in range(bs):
            #batch_mask = coords[:, 0]==b
            points, image = inputs['points'][b][:,:3], inputs['img'][b]
            uvs, depth, filtered_mask = project_pcl_to_image(
                points,
                inputs['extrinsic'][b][0],
                inputs['K'][b][0],
                image.shape[-2:],
                True
            )

            colors = np.zeros([points.shape[0], 3])
            colors[filtered_mask.cpu().numpy()] = image[:, uvs[:, 1], uvs[:, 0]].T.cpu().numpy()
            points = BasicPointCloud(points=points.cpu().numpy(), colors=colors, normals=np.zeros([points.shape[0], 3]))
            self.gaussian.create_from_pcd(points, 1)

            # scale, rot, opacity(optional, 0 or 1)
            # normal, img_feat, pts_feat -> rot, scale

            K, extrinsic = inputs['K'][b], inputs['extrinsic'][b]
            focal_length_x, focal_length_y = K[..., 0, 0], K[..., 1, 1]
            R, T = extrinsic[0, :3, :3].T, extrinsic[0, :3, -1]
            height, width = image.shape[-2], image.shape[-1]
            cam_info = dataset_readers.CameraInfo(uid=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FovX=focal2fov(focal_length_x, width), FovY=focal2fov(focal_length_y, height),
                                                image=image, image_path=None, image_name='image', width=width, height=height)
            cam = cameras.Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=image, gt_alpha_mask=None,
                    image_name=cam_info.image_name, uid=0, data_device=image.device)
            
            # render
            # point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
            # max_depth = point_depth.max()
            bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            res = render(cam, self.gaussian, PipelineParams(), bg)

            debug=True
            if debug:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(image.permute(1,2,0).cpu().numpy())
                plt.scatter(uvs[:, 0].cpu().numpy(), uvs[:, 1].cpu().numpy(), c=depth.cpu().numpy(), s=4, cmap='jet_r')
                plt.savefig('z_projpoint.png')

                plt.figure()
                plt.imshow(res['render'].detach().permute(1,2,0).cpu().numpy())
                plt.savefig('z_render.png')

                pts = points.points
                points_vis = np.zeros((pts.shape[0], 6))
                points_vis[:, :3] = pts
                points_vis[:, 3:] = colors*255
                points_vis.astype(np.float32).tofile('z.bin')

                self.gaussian.save_ply('z.ply')


        return rendered_img
import os
import cv2
import argparse
import numpy as np
from airobot import Robot, log_info
from airobot.utils.common import euler2quat, quat2euler

class PokingEnv(object):
    """
    Poking environment: loads initial object on a table
    """
    def __init__(self, ifRender=False):
        np.set_printoptions(precision=3, suppress=True)
        # table scaling and table center location
        self.table_scaling = 0.6 # tabletop x ~ (0.3, 0.9); y ~ (-0.45, 0.45)
        self.table_x = 0.6
        self.table_y = 0
        self.table_z = 0.6
        self.table_surface_height = 0.975 # get from running robot.cam.get_pix.3dpt
        self.table_ori = euler2quat([0, 0, np.pi/2])
        # task space x ~ (0.4, 0.8); y ~ (-0.3, 0.3)
        self.max_arm_reach = 0.91
        self.workspace_max_x = 0.75 # 0.8 discouraged, as box can move past max arm reach 
        self.workspace_min_x = 0.4
        self.workspace_max_y = 0.3
        self.workspace_min_y = -0.3
        # robot end-effector
        self.ee_min_height = 0.99
        self.ee_rest_height = 1.1 # stick scale="0.0001 0.0001 0.0007"
        self.ee_home = [self.table_x, self.table_y, self.ee_rest_height]
        # initial object location
        self.box_z = 1 - 0.005
        self.box_pos = [self.table_x, self.table_y, self.box_z]
        self.box_size = 0.02 # distance between center frame and size, box size 0.04
        # poke config: poke_len by default [0.06-0.04]
        self.poke_len_min = 0.06 # 0.06 ensures no contact with box empiracally
        self.poke_len_range = 0.04
        # image processing config
        self.row_min = 40
        self.row_max = 360
        self.col_min = 0
        self.col_max = 640
        self.output_row = 100
        self.output_col = 200

        # load robot
        self.robot = Robot('ur5e_stick', pb_render=ifRender)
        self.robot.arm.go_home()
        self.ee_origin = self.robot.arm.get_ee_pose()
        self.go_home()
        self._home_jpos = self.robot.arm.get_jpos()
        # load table
        self.table_id = self.load_table()
        # load box
        self.box_id = self.load_box()
        # initialize camera matrices
        self.robot.cam.setup_camera(focus_pt=[0.7, 0, 1.],
                                    dist=0.5, yaw=90, pitch=-60, roll=0)
        self.ext_mat = self.robot.cam.get_cam_ext()
        self.int_mat = self.robot.cam.get_cam_int()


    def poke(self, save_dir='data/image'):
        # setup directories
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_depth_dir = save_dir + '_depth'
        if not os.path.exists(save_depth_dir):
            os.makedirs(save_depth_dir)
        # poke loop
        self.go_home()
        self.reset_box()
        poke_index = 0
        while True:
            obj_pos, obj_quat, _, lin_vel = self.get_box_pose()
            obj_x, obj_y, obj_z = obj_pos
            # check if cube is on table and still
            if obj_z < self.table_surface_height or lin_vel[0] > 1e-3:
                print("object height: ", obj_z, "object x linear velocity: ", lin_vel[0])
                self.reset_box()
                continue
            # log images
            rgb, dep = self.get_img()  # rgb 0-255 uint8; dep float32
            img = self.resize_rgb(rgb) # img 0-255 float32
            cv2.imwrite(save_dir + '/' + str(poke_index) + '.png',
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) # shape (100,200,3)
            np.save(save_dir + '_depth/' + str(poke_index), dep) # shape (480,640)
            # generate random poke through box center
            poke_ang, poke_len, start_x, start_y, end_x, end_y = self.sample_poke(obj_x, obj_y)
            # calc poke and obj locations in image pixel space
            start_r, start_c = self.xyz2pixel(start_x, start_y)
            end_r, end_c = self.xyz2pixel(end_x, end_y)
            obj_r, obj_c = self.xyz2pixel(obj_x, obj_y)
            # execute poke
            start_jpos, end_jpos = self.execute_poke(start_x, start_y, end_x, end_y)
            js1, js2, js3, js4, js5, js6 = start_jpos
            je1, je2, je3, je4, je5, je6 = end_jpos
            # log poke
            with open(save_dir + '.txt', 'a') as file:
                file.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                    (poke_index, start_x, start_y, end_x, end_y, obj_x, obj_y,
                    obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3],
                    js1, js2, js3, js4, js5, js6, je1, je2, je3, je4, je5, je6,
                    start_r, start_c, end_r, end_c, obj_r, obj_c, poke_ang, poke_len))
            # log num of pokes
            if poke_index % 1000 == 0:
                log_info('number of pokes: %sk' % str(poke_index/1000))
            poke_index += 1



    def move_ee_xyz(self, delta_xyz):
        return self.robot.arm.move_ee_xyz(delta_xyz, eef_step=0.015)


    def set_ee_pose(self, pos, ori=None, ignore_physics=False):
        jpos = self.robot.arm.compute_ik(pos, ori)
        return self.robot.arm.set_jpos(jpos, wait=True, ignore_physics=ignore_physics)


    def go_home(self):
        self.set_ee_pose(self.ee_home, self.ee_origin[1])


    def tele_home(self):
        # directly use joint values as solving ik may return different values
        return self.robot.arm.set_jpos(position=self._home_jpos, ignore_physics=True)


    def load_table(self):
        return self.robot.pb_client.load_urdf('table/table.urdf',
                                              [self.table_x, self.table_y, self.table_z],
                                               self.table_ori,
                                               scaling=self.table_scaling)


    def load_box(self, pos=None, quat=None, rgba=[1, 0, 0, 1]):
        if pos is None:
            pos = self.box_pos
        return self.robot.pb_client.load_geom('box', size=self.box_size,
                                                     mass=1,
                                                     base_pos=pos,
                                                     base_ori=quat,
                                                     rgba=rgba)


    def reset_box(self, box_id=None, pos=None, quat=None):
        if box_id is None:
            box_id = self.box_id
        if pos is None:
            pos = self.box_pos
        return self.robot.pb_client.reset_body(box_id, pos, quat)


    def remove_box(self, box_id):
        self.robot.pb_client.remove_body(box_id)


    def get_ee_pose(self):
        return self.robot.arm.get_ee_pose()


    def get_box_pose(self, box_id=None):
        if box_id is None:
            box_id = self.box_id
        pos, quat, lin_vel, _ = self.robot.pb_client.get_body_state(box_id)
        rpy = quat2euler(quat=quat)
        return pos, quat, rpy, lin_vel


    def get_img(self):
        rgb, depth = self.robot.cam.get_images(get_rgb=True, get_depth=True)
        return rgb, depth


    def resize_rgb(self, rgb):
        img = rgb[self.row_min:self.row_max, self.col_min:self.col_max] # type int64
        resized_img = cv2.resize(img.astype('float32'),
                                  dsize=(self.output_col, self.output_row),
                                  interpolation=cv2.INTER_CUBIC)
        return resized_img


    def xyz2pixel(self, X, Y, Z=None):
        """
        return fractional pixel representations from world frame XYZ coordinates
        """
        if Z is None:
            Z = self.table_surface_height
        if (self.ext_mat is None) or (self.int_mat is None):
            raise ValueError('Please set up the camera matrices')
        XYZ = np.array([X, Y, Z, 1])
        xyz = np.linalg.inv(self.ext_mat).dot(XYZ)[:3]
        pixel_xyz = self.int_mat.dot(xyz)
        pixel_xyz = pixel_xyz / pixel_xyz[2]
        row_scale = (self.row_max - self.row_min) / float(self.output_row)
        col_scale = (self.col_max - self.col_min) / float(self.output_col)
        assert col_scale == row_scale
        pixel_col = (pixel_xyz[0] - self.col_min) / col_scale # due to image cropping and scaling
        pixel_row = (pixel_xyz[1] - self.row_min) / row_scale
        return pixel_row, pixel_col


    def sample_poke(self, obj_x, obj_y):
        while True:
            # choose poke angle along the z axis
            poke_ang = np.random.random() * np.pi * 2 - np.pi
            # choose poke length
            poke_len = np.random.random() * self.poke_len_range + self.poke_len_min
            # calc starting poke location and ending poke loaction
            start_x = obj_x - self.poke_len_min * np.cos(poke_ang)
            start_y = obj_y - self.poke_len_min * np.sin(poke_ang)
            end_x = obj_x + poke_len * np.cos(poke_ang)
            end_y = obj_y + poke_len * np.sin(poke_ang)
            start_radius = np.sqrt(start_x**2 + start_y**2)
            end_radius = np.sqrt(end_x**2 + end_y**2)
            # find valid poke that does not lock the arm
            if start_radius < self.max_arm_reach \
                and end_radius + self.poke_len_min < self.max_arm_reach \
                and end_x > self.workspace_min_x and end_x < self.workspace_max_x \
                and end_y > self.workspace_min_y and end_y < self.workspace_max_y:
                # find poke that does not push obj out of workspace (camera view)
                break
        return poke_ang, poke_len, start_x, start_y, end_x, end_y


    def execute_poke(self, start_x, start_y, end_x, end_y):
        # move to starting poke location
        self.move_ee_xyz([start_x-self.ee_home[0], start_y-self.ee_home[1], 0])
        self.move_ee_xyz([0, 0, self.ee_min_height-self.ee_rest_height])
        # log joint angles
        start_jpos = self.robot.arm.get_jpos()
        self.move_ee_xyz([end_x-start_x, end_y-start_y, 0]) # poke
        end_jpos = self.robot.arm.get_jpos()
        # important that we use move_ee_xyz, as set_ee_pose can throw obj in motion
        self.move_ee_xyz([0, 0, self.ee_rest_height-self.ee_min_height])
        # move arm away from camera view
        self.go_home() # important to have one set_ee_pose every loop to reset accu errors
        return start_jpos, end_jpos


    # def tele_poke(self, start_x, start_y, end_x, end_y):
    #     # teleport to poke locations
    #     self.set_ee_pose(pos=[start_x, start_y, self.ee_min_height], ignore_physics=True)
    #     # log joint angles
    #     start_jpos = self.robot.arm.get_jpos()
    #     self.move_ee_xyz([end_x-start_x, end_y-start_y, 0]) # poke
    #     end_jpos = self.robot.arm.get_jpos()
    #     self.tele_home()
    #     return start_jpos, end_jpos


    def stress_test_poke(self, obj_x=None, obj_y=None):
        if obj_x is None:
            obj_x = self.workspace_max_x
        if obj_y is None:
            obj_y = self.workspace_max_y
        while True:
            # move the object to the cornner of workspace
            self.reset_box(pos=[obj_x, obj_y, self.box_z])
            obj_x, obj_y, obj_z = self.get_box_pose()[0]
            pa, pl, sx, sy, ex, ey = self.sample_poke(obj_x, obj_y)
            print(sx, sy)
            self.execute_poke(sx, sy, ex, ey)
            _, _ = self.get_img()

    def boundary_poke(self, length=0.9):
        # 0.91 is the max length without locking the arm
        interval = np.pi/6 / 6
        for i in range(7):
            ang = -i * interval
            x = length * np.cos(ang)
            y = length * np.sin(ang)
            self.move_ee_xyz([x-0.6, y-0, 0])
            self.move_ee_xyz([0, 0, -0.1])
            self.move_ee_xyz([0, 0, 0.1])
            self.go_home()


    def reproducibility_test(self, seed=0):
        while True:
            self.go_home()
            self.reset_box()
            np.random.seed(seed)
            _, _, start_x, start_y, end_x, end_y = self.sample_poke(self.box_pos[0], self.box_pos[1])
            self.execute_poke(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
            self.get_img()
            pos, _, rpy, _ = self.get_box_pose()
            print(pos, rpy)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='tag will enable rendering')
    args = parser.parse_args()
    env = PokingEnv(ifRender=args.render)
    from IPython import embed
    embed()

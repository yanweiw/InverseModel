import time
import numpy as np
import matplotlib.pyplot as plt
import airobot as ar
from airobot import Robot
from airobot.utils.common import euler2quat
from airobot.utils.common import quat2euler
from airobot.utils.pb_util import load_urdf
from airobot.utils.pb_util import load_geom
from airobot.utils.pb_util import get_body_state
from airobot.utils.pb_util import reset_body

# under scale factor
table_length = 1.0
table_wideth = 0.6 # but effective arm reach is 0.9
table_scaling = 0.6
arm_span = 0.9
min_height = 1.0
rest_height = 1.1
home = [0.3, 0, rest_height]
table_x = 0.59 # task space x ~ (0.3, 0.9); y ~ (-0.5, 0.5)
table_y = 0
table_z = 0.60
box_pos = [table_x, table_y, 1]
box_size = 0.02
workspace_max_x = 0.8 - 0.05
workspace_min_x = 0.4
workspace_max_y = 0.4 - 0.1
workspace_min_y = -0.4 + 0.1


def main():
    """
    this function loads initial oboject on a table
    """
    np.set_printoptions(precision=3, suppress=True)

    # load robot
    robot = Robot('ur5e', arm_cfg={'render': True, 'self_collision': True})
    robot.arm.go_home()
    origin = robot.arm.get_ee_pose()
    def go_home():
        robot.arm.set_ee_pose(home, origin[1])
        robot.arm.eetool.close()
    # init robot
    go_home()

    # load table
    table_ori = euler2quat([0, 0, np.pi/2.0])
    table_pos = [table_x, table_y, table_z]
    table_id = load_urdf('table/table.urdf', table_pos, table_ori, scaling=table_scaling)

    # load box
    box_id = load_geom('box', size=box_size, mass=1, base_pos=box_pos, rgba=[1, 0, 0, 1])

    # init camera
    def get_img():
        # screenshot camera images
        focus_pt = [0.7, 0, 1.]
        robot.cam.setup_camera(focus_pt=focus_pt, dist=0.5, yaw=90, pitch=-60, roll=0)
        rgb, depth = robot.cam.get_images(get_rgb=True, get_depth=True)

    # test run
    def test():
        time_to_sleep = 0.5
        go_home()
        pose = robot.arm.get_ee_pose()
        robot.arm.set_ee_pose([pose[0][0], pose[0][1], min_height], origin[1])
        time.sleep(time_to_sleep)
        get_img()
        # test boundary
        # robot.arm.set_ee_pose([pose[0][0], pose[0][1]+table_length/2.0, min_height], origin[1])
        # # robot.arm.eetool.close()
        # time.sleep(time_to_sleep)
        # get_img()
        # robot.arm.move_ee_xyz([0, -table_length, 0])
        # time.sleep(time_to_sleep)
        # get_img()
        # pose = robot.arm.get_ee_pose()
        # robot.arm.set_ee_pose([pose[0][0], pose[0][1], rest_height], origin[1])
        # time.sleep(time_to_sleep)
        # get_img()
        # robot.arm.move_ee_xyz([0, table_length, 0])
        # time.sleep(time_to_sleep)
        # get_img()
        # pose = robot.arm.get_ee_pose()
        # robot.arm.set_ee_pose([pose[0][0], pose[0][1], min_height], origin[1])
        # time.sleep(time_to_sleep)
        # get_img()
        # test arc
        for i in list([np.pi/3.0, np.pi*2/5.0, np.pi*3/7.0, np.pi/2.0,
                       np.pi*4/7.0, np.pi*3/5.0, np.pi*2/3.0]):
            robot.arm.set_ee_pose([arm_span*np.sin(i),
            arm_span*np.cos(i), rest_height], origin[1])
            time.sleep(time_to_sleep)
            get_img()
            robot.arm.set_ee_pose([arm_span*np.sin(i),
            arm_span*np.cos(i), min_height], origin[1])
            time.sleep(time_to_sleep)
            get_img()
            robot.arm.set_ee_pose([arm_span*np.sin(i),
            arm_span*np.cos(i), rest_height], origin[1])
            time.sleep(time_to_sleep)
            get_img()


    # # log object
    # pos, quat, lin_vel, ang_vel = get_body_state(box_id)
    # ar.log_info('Box:')
    # ar.log_info('     position: %s' % np.array2string(pos, precision=2))
    # ar.log_info('     quaternion: %s' % np.array2string(quat, precision=2))
    # ar.log_info('     linear vel: %s' % np.array2string(lin_vel, precision=2))
    # ar.log_info('     angular vel: %s' % np.array2string(ang_vel, precision=2))

    # plt.figure()
    # plt.imshow(rgb)
    # plt.figure()
    # plt.imshow(depth * 25, cmap='gray', vmin=0, vmax=255)
    # print('Maximum Depth (m): %f' % np.max(depth))
    # print('Minimum Depth (m): %f' % np.min(depth))
    # plt.show()

    def poke():
        get_img()
        while True:
            obj_pos, obj_quat, _, _ = get_body_state(box_id)
            obj_ang = quat2euler(obj_quat)[2] # -pi ~ pi
            obj_x, obj_y, obj_z = obj_pos
            while True:
                # choose random poke point on the object
                poke_x = np.random.random()*box_size + obj_x - box_size/2.0
                poke_y = np.random.random()*box_size + obj_y - box_size/2.0
                # choose poke angle along the z axis
                poke_ang = np.random.random() * np.pi * 2 - np.pi
                # choose poke length
                poke_len = np.random.random() * 0.05 + 0.05 # [0.05-0.15]
                # calc starting poke location and ending poke loaction
                start_x = poke_x - poke_len * np.cos(poke_ang)
                start_y = poke_y - poke_len * np.sin(poke_ang)
                end_x = poke_x + poke_len * np.cos(poke_ang)
                end_y = poke_y + poke_len * np.sin(poke_ang)
                if end_x > workspace_min_x and end_x < workspace_max_x \
                    and end_y > workspace_min_y and end_y < workspace_max_y:
                    break
            robot.arm.set_ee_pose([start_x, start_y, rest_height], origin[1])
            # get_img()
            # time.sleep(0.5)
            robot.arm.move_ee_xyz([0, 0, min_height-rest_height])
            robot.arm.move_ee_xyz([end_x-start_x, end_y-start_y, 0])
            robot.arm.move_ee_xyz([0, 0, rest_height-min_height])
            get_img()


    from IPython import embed
    embed()

    # time.sleep(10)

if __name__ == '__main__':
    main()

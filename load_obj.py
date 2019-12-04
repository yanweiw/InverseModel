import time
import numpy as np
import matplotlib.pyplot as plt
import airobot as ar
from airobot import Robot
from airobot.utils.common import euler2quat
from airobot.utils.pb_util import load_urdf
from airobot.utils.pb_util import load_geom
from airobot.utils.pb_util import get_body_state

# under scale factor
table_length = 1.0
table_wideth = 0.6 # but effective arm reach is 0.9
table_scaling = 0.6
arm_span = 0.9
min_height = 1.0
rest_height = 1.3
home = [0.3, 0, rest_height]
table_x = 0.59
table_y = 0
table_z = 0.60

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
    box_pos = [table_x, table_y, 1]
    box_id = load_geom('box', size=0.02, mass=1, base_pos=box_pos, rgba=[1, 0, 0, 1])

    # init camera
    def get_img():
        # screenshot camera images
        focus_pt = [0.5, 0, 0]
        robot.cam.setup_camera(focus_pt=focus_pt, dist=2.2, yaw=90, pitch=-90, roll=0)
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
        robot.arm.set_ee_pose([pose[0][0], pose[0][1]+table_length/2.0, min_height], origin[1])
        # robot.arm.eetool.close()
        time.sleep(time_to_sleep)
        get_img()
        robot.arm.move_ee_xyz([0, -table_length, 0])
        time.sleep(time_to_sleep)
        get_img()
        pose = robot.arm.get_ee_pose()
        robot.arm.set_ee_pose([pose[0][0], pose[0][1], rest_height], origin[1])
        time.sleep(time_to_sleep)
        get_img()
        robot.arm.move_ee_xyz([0, table_length, 0])
        time.sleep(time_to_sleep)
        get_img()
        pose = robot.arm.get_ee_pose()
        robot.arm.set_ee_pose([pose[0][0], pose[0][1], min_height], origin[1])
        time.sleep(time_to_sleep)
        get_img()
        # test arc
        for i in list([np.pi/3.0, np.pi/2.5, np.pi/2.0,
                    np.pi*3/5.0, np.pi*2/3.0]):
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

    from IPython import embed
    embed()

    # time.sleep(10)

if __name__ == '__main__':
    main()

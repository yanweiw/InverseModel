import time
import numpy as np
import matplotlib.pyplot as plt
import airobot as ar
from airobot import Robot
from airobot.utils.common import euler2quat
from airobot.utils.pb_util import load_urdf
from airobot.utils.pb_util import load_geom
from airobot.utils.pb_util import get_body_state

def main():
    """
    this function loads initial oboject on a table
    """
    np.set_printoptions(precision=3, suppress=True)
    robot = Robot('ur5e', arm_cfg={'render': True, 'self_collision': True})
    robot.arm.go_home()
    robot.arm.eetool.close()
    robot.arm.move_ee_xyz([-0.4, 0, 0])

    # table orientation
    table_ori = euler2quat([0, 0, np.pi/2])
    table_id = load_urdf('table/table.urdf', [0.5, 0, 0.4], table_ori, scaling=1)

    # loading box
    box_id = load_geom('box', size=0.02, mass=1, base_pos=[0.5, 0, 1], rgba=[1, 0, 0, 1])

    # log object
    pos, quat, lin_vel, ang_vel = get_body_state(box_id)
    ar.log_info('Box:')
    ar.log_info('     position: %s' % np.array2string(pos, precision=2))
    ar.log_info('     quaternion: %s' % np.array2string(quat, precision=2))
    ar.log_info('     linear vel: %s' % np.array2string(lin_vel, precision=2))
    ar.log_info('     angular vel: %s' % np.array2string(ang_vel, precision=2))

    # screenshot camera images
    focus_pt = [0.5, 0, 0]
    robot.cam.setup_camera(focus_pt=focus_pt,
                           dist=2.2,
                           yaw=90,
                           pitch=-90,
                           roll=0)
    rgb, depth = robot.cam.get_images(get_rgb=True,
                                      get_depth=True)
    plt.figure()
    plt.imshow(rgb)
    plt.figure()
    plt.imshow(depth * 25, cmap='gray', vmin=0, vmax=255)
    print('Maximum Depth (m): %f' % np.max(depth))
    print('Minimum Depth (m): %f' % np.min(depth))
    plt.show()

    # time.sleep(10)

if __name__ == '__main__':
    main()

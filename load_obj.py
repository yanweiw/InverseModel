import os
import argparse
import time
import cv2
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
import pybullet as p
from airobot.utils.pb_util import remove_body

# under scale factor
table_length = 1.0
table_wideth = 0.6 # but effective arm reach is 0.9
table_scaling = 0.6
arm_span = 0.9
min_height = 1.0 #+ 0.02
rest_height = 1.1
home = [0.6, 0, rest_height]
table_x = 0.59 # task space x ~ (0.4, 0.75); y ~ (-0.3, 0.3)
table_y = 0
table_z = 0.60
box_z = 1.0
box_pos = [table_x, table_y, box_z]
box_size = 0.02
workspace_max_x = 0.8 - 0.05
workspace_min_x = 0.4
workspace_max_y = 0.4 - 0.1
workspace_min_y = -0.4 + 0.1
poke_len_mean=0.06 # poke_len by default [0.06-0.1]
poke_len_std=0.04


def main(ifRender=False):
    """
    this function loads initial oboject on a table,
    """
    np.set_printoptions(precision=3, suppress=True)

    # load robot
    robot = Robot('ur5e', arm_cfg={'render': ifRender, 'self_collision': True})
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
        # crop the rgb
        img = rgb[40:360, 0:640]
        # low pass filter : Gaussian blur
        # blurred_img = cv2.GaussianBlur(img.astype('float32'), (5, 5), 0)
        small_img = cv2.resize(img.astype('float32'), dsize=(200, 100),
                    interpolation=cv2.INTER_CUBIC) # numpy array dtype numpy int64 by default
        return small_img

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

    def poke(save_dir='image'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        index = 0
        curr_img = get_img()
        while True:
            obj_pos, obj_quat, lin_vel, _ = get_body_state(box_id)
            obj_ang = quat2euler(obj_quat)[2] # -pi ~ pi
            obj_x, obj_y, obj_z = obj_pos
            # check if cube is on table
            if obj_z < box_z / 2.0 or lin_vel[0] > 1e-3: # important that box is still
                print(obj_z, lin_vel[0])
                reset_body(box_id, box_pos)
                continue
            while True:
                # choose random poke point on the object
                poke_x = np.random.random()*box_size + obj_x - box_size/2.0
                poke_y = np.random.random()*box_size + obj_y - box_size/2.0
                # choose poke angle along the z axis
                poke_ang = np.random.random() * np.pi * 2 - np.pi
                # choose poke length
                poke_len = np.random.random() * poke_len_std + poke_len_mean
                # calc starting poke location and ending poke loaction
                start_x = poke_x - poke_len * np.cos(poke_ang)
                start_y = poke_y - poke_len * np.sin(poke_ang)
                end_x = poke_x + poke_len * np.cos(poke_ang)
                end_y = poke_y + poke_len * np.sin(poke_ang)
                if end_x > workspace_min_x and end_x < workspace_max_x \
                    and end_y > workspace_min_y and end_y < workspace_max_y:
                    break
            robot.arm.move_ee_xyz([start_x-home[0], start_y-home[1], 0], 0.015)
            robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
            js1, js2, js3, js4, js5, js6 = robot.arm.get_jpos()
            robot.arm.move_ee_xyz([end_x-start_x, end_y-start_y, 0], 0.015)
            je1, je2, je3, je4, je5, je6 = robot.arm.get_jpos()
            # important that we use move_ee_xyz, as set_ee_pose can throw obj in motion
            robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
            # move arm away from camera view
            go_home() # important to have one set_ee_pose every loop to reset accu errors
            next_img = get_img()
            with open(save_dir + '.txt', 'a') as file:
                file.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                    (index, start_x, start_y, end_x, end_y, obj_x, obj_y,
                    obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3],
                    js1, js2, js3, js4, js5, js6, je1, je2, je3, je4, je5, je6))
            cv2.imwrite(save_dir + '/' + str(index) +'.png',
                        cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
            curr_img = next_img
            if index % 1000 == 0:
                ar.log_info('number of pokes: %sk' % str(index/1000))
            index += 1

    def test_data(filename):
        data = np.loadtxt(filename, unpack=True)
        idx, stx, sty, edx, edy, obx, oby, qt1, qt2, qt3, qt4, js1, js2, js3, js4, js5, js6, je1, je2, je3, je4, je5, je6 = data
        go_home()
        reset_body(box_id, box_pos)
        _ = get_img()
        for i in range(5):
            robot.arm.set_ee_pose([stx[i], sty[i], rest_height], origin[1])
            robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
            robot.arm.move_ee_xyz([edx[i]-stx[i], edy[i]-sty[i], 0], 0.015)
            robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
            _ = get_img()
            start_jpos = robot.arm.compute_ik([stx[i], sty[i], min_height])
            end_jpos = robot.arm.compute_ik([edx[i], edy[i], min_height])
            print('start')
            print(js1[i], js2[i], js3[i], js4[i], js5[i], js6[i])
            print(start_jpos)
            print('end')
            print(je1[i], je2[i], je3[i], je4[i], je5[i], je6[i])
            print(end_jpos)
            time.sleep(1)

    def eval_poke(ground_truth, img_idx, poke):
        # ground_truth is the entire matrix from 'image_xx.txt' file
        # img_idx is image index, first column of ground_truth
        # poke is the predicted 4 vector
        gt = ground_truth[img_idx:img_idx+2]
        tgt_posi = gt[1, 5:8]
        tgt_posi[-1] = min_height
        tgt_quat = gt[1, 7:11]
        init_posi = gt[0, 5:8]
        init_posi[-1] = min_height
        init_quat = gt[0, 7:11]
        gt_poke = gt[0, 1:5]
        go_home()
        reset_body(box_id, init_posi, init_quat)

        box_id2 = load_geom('box', size=box_size, mass=1,
            base_pos=tgt_posi, base_ori=tgt_quat, rgba=[0,0,1,0.5])
        p.setCollisionFilterPair(box_id, box_id2, -1, -1, enableCollision=0)
        # p.setCollisionFilterPair(22, 12, -1, -1, enableCollision=0)
        _ = get_img()
        # time.sleep(1)
        # reset_body(box_id, init_posi, init_quat)
        # robot.arm.move_ee_xyz([gt_poke[0]-home[0], gt_poke[1]-home[1], 0], 0.015)
        # robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
        # robot.arm.move_ee_xyz([gt_poke[2]-gt_poke[0], gt_poke[3]-gt_poke[1], 0], 0.015)
        # robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
        # go_home()
        # _ = get_img()
        # reset_body(box_id, init_posi, init_quat)
        robot.arm.move_ee_xyz([poke[0]-home[0], poke[1]-home[1], 0], 0.015)
        robot.arm.move_ee_xyz([0, 0, min_height-rest_height], 0.015)
        robot.arm.move_ee_xyz([poke[2]-poke[0], poke[3]-poke[1], 0], 0.015)
        robot.arm.move_ee_xyz([0, 0, rest_height-min_height], 0.015)
        go_home()
        _ = get_img()
        curr_posi, _, _, _ = get_body_state(box_id)
        dist = np.sqrt((tgt_posi[0]-curr_posi[0])**2 + (tgt_posi[1]-curr_posi[1])**2)
        print(dist)
        # time.sleep(0.5)
        remove_body(box_id2)
        return dist

    def calc_dist(gtfile, pdfile):
        # this function generate 50 visualization sequence
        true = np.loadtxt(gtfile)
        pred = np.loadtxt(pdfile)
        accu_dist = 0.0
        for i in range(50):
            accu_dist += eval_poke(true, int(pred[i][0]), pred[i][1:5])
        return accu_dist / 50.0


    from IPython import embed
    embed()

    # time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # omit argument to produce no rendering
    parser.add_argument('--render', type=bool, help='if rendering')
    args = parser.parse_args()
    main(ifRender=args.render)

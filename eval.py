import numpy as np
import pybullet as p
import argparse
from environment import PokingEnv


# label index in the poke data array (31, 1)
# index 0 is the poke index corresponding to starting image
stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 # jpos after poke
sr, stc, edr, edc, obr, obc = 23, 24, 25, 26, 27, 28 # row and col locations in image
pang, plen = 29, 30 # poke angle and poke length

class EvalPoke():
    """
    Calculating poking statistics for trained models
    """
    def __init__(self, ifRender=False):
        self.env = PokingEnv(ifRender)
        self.gt = None # ground_truth
        self.pd = None # prediction
        self.box_id2 = None # goal box id


    def load_pokedata(self, gtfile, pdfile=None):
        """
        loading ground_truth and predictions
        """
        self.gt = np.loadtxt(gtfile)
        if pdfile is not None:
            self.pd = np.loadtxt(pdfile)


    def validate_data(self, num_to_see=None, tag='world'):
        """
        validate that data is consistent: obj_start pose + poke -> obj_end pose
        visualize ground truth pokes and predicted pokes
        tag: world, wpoke, joint, pixel, default None correspond to ground truth
        """
        if self.gt is None:
            raise ValueError('load ground truth')
        if self.box_id2 is not None:
            self.env.remove_box(self.box_id2)
        if num_to_see is None:
            query = range(len(self.gt) - 1) # query the starting obj pose
        else:
            query = np.random.randint(0, len(self.gt)-1, num_to_see)
        for i in query:
            poke = self.gt[i, stx:edy+1]
            obj_start = self.gt[i, obx:qt4+1]
            obj_end = self.gt[i+1, obx:qt4+1]
            if tag == 'world': # ground truth pokes in world frame
                pass
            elif tag == 'wpoke': # gt pokes calc from poke angle and length
                poke_ang, poke_len = self.gt[i, pang], self.gt[i, plen]
                poke = self.wpoke2world(obj_start[0], obj_start[1], poke_ang, poke_len)
            elif tag == 'pixel': # gt pokes calc from pixels and depth images
                pass
            elif tag == 'joint': # gt pokes calc from joint values
                pass
            else:
                raise ValueError('tag has to be world, wpoke, pixel or joint')

            _ = self.eval_poke(poke, obj_start, obj_end)


    def eval_poke(self, poke, obj_start, obj_end):
        """
        evaluate/visualize a single poke
        obj_start and obj_end: (x, y, q1, q2, q3, q4)
        """
        self.env.reset_box(pos=[obj_start[0], obj_start[1], self.env.box_z],
                      quat=obj_start[-4:])
        self.set_goal(pos=[obj_end[0], obj_end[1], self.env.box_z],
                      quat=obj_end[-4:])
        _, _ = self.env.execute_poke(poke[0], poke[1], poke[2], poke[3])
        _, _ = self.env.get_img()
        curr_pos, _, _, _ = self.env.get_box_pose()
        dist = np.sqrt((curr_pos[0]-obj_end[0])**2 + (curr_pos[1]-obj_end[1])**2)
        # remove target box
        self.env.remove_box(self.box_id2)
        self.box_id2 = None
        print(dist)
        return dist


    def set_goal(self, pos, quat):
        self.box_id2 = self.env.load_box(pos=pos, quat=quat, rgba=[0,0, 1,0.5])
        p.setCollisionFilterPair(self.env.box_id, self.box_id2, -1, -1, enableCollision=0)
        p.setCollisionFilterPair(self.box_id2, 1, -1, 10, enableCollision=0) # with arm
        # _, _ = self.env.get_img()


    def batch_eval(self, gtfile, pdfile, tag, test_num=100):
        self.load_pokedata(gtfile, pdfile)
        if (self.gt is None) or (self.pd is None):
            raise ValueError('load ground truth and prediction file')
        accu_dist = 0.0
        query = np.random.randint(0, len(self.pd)-1, test_num) # len-1 avoids end row access
        for i in query:
            img_idx = int(self.pd[i][0])
            obj_start = self.gt[img_idx, obx:qt4+1]
            obj_end = self.gt[img_idx+1, obx:qt4+1]
            # pred data comes in [indices, pred_pokes, true_pokes]
            if tag == 'world':
                poke = self.pd[i][1:5]
            elif tag == 'wpoke':
                poke_ang = self.pd[i][3]
                poke_len = self.pd[i][4]
                print('ang: ', poke_ang, ' len: ', poke_len)
                poke = self.wpoke2world(obj_start[0], obj_start[1], poke_ang, poke_len)
            accu_dist += self.eval_poke(poke, obj_start, obj_end)
        return accu_dist / len(query)


    def wpoke2world(self, obj_x, obj_y, poke_ang, poke_len):
        """
        take obj position, poke angle and poke length to return poke in world frame
        """
        start_x = obj_x - self.env.poke_len_min * np.cos(poke_ang)
        start_y = obj_y - self.env.poke_len_min * np.sin(poke_ang)
        end_x = obj_x + poke_len * np.cos(poke_ang)
        end_y = obj_y + poke_len * np.sin(poke_ang)
        return [start_x, start_y, end_x, end_y]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='enables rendering')
    parser.add_argument('--gt', default=None, help='gt file')
    parser.add_argument('--pd', default=None, help='pred file')
    parser.add_argument('--tag', required=True, help='world, wpoke, joint or pixel')
    parser.add_argument('--num', type=int, default=100, help='test num')
    args = parser.parse_args()
    eva = EvalPoke(ifRender=args.render)

    if (args.gt is not None) and (args.pd is not None):
        print(eva.batch_eval(args.gt, args.pd, args.tag, args.num))

    from IPython import embed
    embed()











#

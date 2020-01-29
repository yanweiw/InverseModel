import numpy as np
import pybullet as p
from environment import PokingEnv


# label index in the poke data array (31, 1)
# index 0 is the poke index corresponding to starting image
stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 # jpos after poke
sr, stc, edr, edc, obr, obc = 23, 24, 25, 26, 27, 28 # row and col locations in image
pang, plen = 29, 30 # poke angle and poke length

class PokeStats:
    """
    Calculating poking statistics for trained models
    """
    def __init__(self, ifRender=True):
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


    def visualize_gt(self, num_to_see=None):
        """
        visualize ground truth pokes
        """
        if self.gt is None:
            raise ValueError('load ground truth')
        if self.box_id2 is not None:
            self.env.remove_box(self.box_id2)
        if num_to_see is None:
            query = range(len(self.gt) - 1) # query the starting obj pose
        else:
            query = np.random.sample(range(len(self.gt) - 1), num_to_see)
        for i in query:
            poke = self.gt[i, stx:edy+1]
            obj_start = self.gt[i, obx:qt4+1]
            obj_end = self.gt[i+1, obx:qt4+1]
            self.visualize_poke(poke, obj_start, obj_end)


    def visualize_poke(self, poke, obj_start, obj_end):
        self.env.reset_box(pos=[obj_start[0], obj_start[1], self.env.box_z],
                      quat=obj_start[-4:])
        self.set_goal(pos=[obj_end[0], obj_end[1], self.env.box_z],
                      quat=obj_end[-4:])
        _, _ = self.env.execute_poke(poke[0], poke[1], poke[2], poke[3])
        _, _ = self.env.get_img()
        self.env.remove_box(self.box_id2)


    def set_goal(self, pos, quat):
        self.box_id2 = self.env.load_box(pos=pos, quat=quat, rgba=[0,0,1,0.5])
        p.setCollisionFilterPair(self.env.box_id, self.box_id2, -1, -1, enableCollision=0)
        p.setCollisionFilterPair(self.box_id2, 1, -1, 10, enableCollision=0) # with arm
        _, _ = self.env.get_img()















#

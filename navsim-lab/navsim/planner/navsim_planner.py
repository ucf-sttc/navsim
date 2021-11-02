import skimage.morphology
import math

from .fmm_planner import FMMPlanner

class NavsimPlanner():

    def __init__(self, env):
        self.navigable_map = env.get_navigable_map().T
        self.goal_map_x, self.goal_map_y = env.unity_to_navmap_location(env.goal_position[0], env.goal_position[2])
        self.turn_angle = 20.0
        self.turn_angle_con = 1.0
        self.last_action = None
        self.env = env

        navigable_map = 1 - self.navigable_map
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(navigable_map, selem) != True
        traversible = traversible.astype(int)
        self.planner = FMMPlanner(traversible)
        self.planner.set_goal((self.goal_map_x, self.goal_map_y))

    @staticmethod
    def map_action(a):
        if a == 3:
            return [1.0, -1.0, -1.0]
        if a == 2:
            return [1.0, 1.0, -1.0]
        if a == 1:
            return [1.0, 0.0, -1.0]
        if a == 0:
            return None

    def plan(self, obs):
        cur_map_x, cur_map_y = self.env.unity_to_navmap_location(obs[-1][0], obs[-1][2])
        cur_map_d1, cur_map_d2 = self.env.unity_to_navmap_rotation(obs[-1][6:10])
        cur_map_o = math.degrees(math.atan2(cur_map_d1, cur_map_d2))
        stg_x, stg_y, _, stop = self.planner.get_short_term_goal((cur_map_x, cur_map_y))
        relative_angle = None
        if stop:
            self.last_action = 0
        else:
            cur_map_o = cur_map_o % 360.0
            if cur_map_o > 180.0:
                cur_map_o -= 360.0

            angle_st_goal = math.degrees(math.atan2(stg_x - cur_map_x, stg_y - cur_map_y))
            relative_angle = (cur_map_o - angle_st_goal) % 360.0
            if relative_angle > 180.0:
                relative_angle -= 360.0
            if relative_angle > self.turn_angle or (relative_angle > self.turn_angle_con and self.last_action != None and self.last_action >= 2):
                self.last_action = 3
            elif relative_angle < -self.turn_angle or (relative_angle < -self.turn_angle_con and self.last_action != None and self.last_action >= 2):
                self.last_action = 2
            else:
                self.last_action = 1
        return NavsimPlanner.map_action(self.last_action)


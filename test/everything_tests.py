# Run with
#DISPLAY=:0.1 pytest everything_tests.py -s
#DISPLAY=:0.1 pytest everything_tests.py -s -k 'test_name'
from navsim.util.dict import ObjDict
from pathlib import Path
import gym
import numpy as np
import navsim
import navsim_envs


class TestEverything():
    run_base_folder = Path(".")
    run_base_folder_str = str(run_base_folder.resolve())
    env_conf = ObjDict({
        "log_folder": str(run_base_folder / "env_log"),
        "env_path": "/data/work/unity-envs/Build2.10.2/Berlin_Walk_V2.x86_64",
        "worker_id": 0,
        "base_port": 5005,
        "seed": 1,
        "timeout": 120,
        "obs_mode": 0,
        "obs_height": 128,
        "obs_width": 128,
        "segmentation_mode": 0,
        "episode_max_steps": 1000,
        "task": 0,
        "goal": 0,
        "goal_distance": 10,
        "traffic_vehicles": 0,
        "agent_car_physics": 0,
        "reward_for_goal": float(1),
        "reward_for_no_viable_path": float(-1),
        "reward_step_mul": float(1),
        "reward_collision_mul": float(1),
        "reward_spl_delta_mul": float(1),
        "env_gpu_id": int(0),
        "debug": False,
        "save_actions": False,
        "save_vector_obs": False,
        "save_visual_obs": False,
        "show_visual": False 
    })

    #Tests whether the navigable map returned has navigable points
    def test_navigable_map(self):
        print("========")
        env = gym.make("navsim-v0", env_config=self.env_conf)
        navigable_map = env.get_navigable_map()
        print(type(navigable_map))
        print(navigable_map)
        print(navigable_map.sum())
        assert navigable_map.sum() > 1
        env.close()
        del env
        


    #Tests whether the return sample points are navigable accouting to the returned navigable map
    def test_sample_navigable_point(self):
        print("========")
        env = gym.make("navsim-v0", env_config=self.env_conf)
        resolution_x=256
        resolution_y=256
        navigable_map = env.get_navigable_map(resolution_x=resolution_x, resolution_y=resolution_y)

        for i in range(0, 30000):
            sampled_point=env.sample_navigable_point(resolution_x=resolution_x, resolution_y=resolution_y)
            map_point=navigable_map[sampled_point[0]][sampled_point[1]]
            print(sampled_point, map_point)
            assert  map_point == 1
        env.close()
        del env

    #Tests whether the navmap to unity and back coordinate conversion are correct 
    def test_coordinate_conversion(self):
        print("========")
        env = gym.make("navsim-v0", env_config=self.env_conf)
        navmap_max_x=256
        navmap_max_y=256

        resolution_margin=15

        unity_loc_original = (0,0)
        navmap_loc = env.unity_loc_to_navmap_loc(*unity_loc_original, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_loc_to_unity_loc(*navmap_loc, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y, navmap_cell_center=False)
        print("unity to nav to unity")
        print(unity_loc_original, navmap_loc, unity_loc)
        assert navmap_loc == (0, 255)
        assert unity_loc == (0, 0)

        unity_loc_original = (0,2665)
        navmap_loc = env.unity_loc_to_navmap_loc(*unity_loc_original, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_loc_to_unity_loc(*navmap_loc, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y,  navmap_cell_center=False)
        print("unity to nav to unity")
        print(unity_loc_original, navmap_loc, unity_loc)
        assert navmap_loc == (0, 0)
        assert unity_loc[0] == 0 and (unity_loc[1] > 2665 - resolution_margin and unity_loc[1] < 2665 + resolution_margin)

        unity_loc_original = (3283,0)
        navmap_loc = env.unity_loc_to_navmap_loc(*unity_loc_original, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_loc_to_unity_loc(*navmap_loc, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y,  navmap_cell_center=False)
        print("unity to nav to unity")
        print(unity_loc_original, navmap_loc, unity_loc)
        assert navmap_loc == (255, 255)
        assert (unity_loc[0] > 3283- resolution_margin and unity_loc[0] < 3283 + resolution_margin) and unity_loc[1] == 0

        unity_loc_original = (3283,2665)
        navmap_loc = env.unity_loc_to_navmap_loc(*unity_loc_original, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_loc_to_unity_loc(*navmap_loc, navmap_max_x=navmap_max_x, navmap_max_y=navmap_max_y,  navmap_cell_center=False)
        print("unity to nav to unity")
        print(unity_loc_original, navmap_loc, unity_loc)
        assert navmap_loc == (255, 0)
        assert (unity_loc[0] > 3283- resolution_margin and unity_loc[0] < 3283 + resolution_margin) and (unity_loc[1] > 2665 - resolution_margin and unity_loc[1] < 2665 + resolution_margin)

        env.close()
        del env

    #Tests whether mnutiple runs produces the same observation values
    def test_reproducibility(self):
        print("========")
        throttle=1
        obs_arr = []
        for outer in range(0,3):
            env = gym.make("navsim-v0", env_config=self.env_conf)
            steering=0
            obs_inner_arr = []

            o = env.reset()
            print(o)
            obs_inner_arr.append(o)
            
            for x in range(0,10):
                o, r, done, i = env.step([throttle,steering,-1])
                obs_inner_arr.append(o)
                print(o)

            obs_arr.append(obs_inner_arr)
            env.close()
            del env

        for i in range(1,len(obs_arr)):
            assert np.array_equal(obs_arr[i-1], obs_arr[i])
            #assert np.isclose(obs_arr[i-1], obs_arr[i], atol=1.0)


    #Tests whether the agent rotates left/right give positive and negative steering
    def test_rotation(self):
        print("========")
        for steering in [-1, 1]:
            throttle=1
            env = gym.make("navsim-v0", env_config=self.env_conf)

            print("Steering is:", steering)
            rot_arr = []
            o = env.reset()
            print("Rot Y in euler", "Rot Y in quaternion", sep=", ")
            for x in range(0,10):
                o, r, done, i = env.step([throttle,steering,-1])
                rot = o[0][7]
                print(env.agent_rotation_in_euler[2], rot, sep=", ")
                rot_arr.append((env.agent_rotation_in_euler[1], rot))

            env.close()
            del env

            for i in range(1,len(rot_arr)):
                rot_diff=(rot_arr[i][1]-rot_arr[i-1][1])
                #euler_rot_diff=(rot_arr[i][0]-rot_arr[i-1][0])
                #Turning right
                if steering > 0:
                    assert rot_diff > 0 
                else:
                    assert rot_diff <= 0 
        

    #Tests whether the agent moves in the correct forward/backward directions given positive and negative throttle 
    def test_throttle(self):
        print("========")
        position_diff_arr = []
        for throttle in [0.1, 0.5, 1]:
            env = gym.make("navsim-v0", env_config=self.env_conf)
            steering=0

            print("Throttle is:", throttle)
            position_arr = []
            o = env.reset()
            for x in range(0,10):
                o, r, done, i = env.step([throttle,steering,-1])
                pos_now=o[0][:3]
                #print(pos_now)
                position_arr.append(pos_now)

            env.close()
            del env

            avg_position_diff=0
            for i in range(1, len(position_arr)):
                position_diff=np.linalg.norm(position_arr[i]-position_arr[i-1])
                print(position_diff)
                avg_position_diff+=position_diff
                if throttle > 0:
                    assert position_diff > 0
                else:
                    assert position_diff <= 0
            avg_position_diff/=len(position_arr)
            position_diff_arr.append(avg_position_diff)


        print(position_diff_arr)




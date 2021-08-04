# `navsim_envs` Package Tutorial

`navsim_envs` is a package that wraps the environments for Reinforcement 
Learning. Current implementation has the following sims encapsulated:
* Arora from UCF

## Installation

`pip install --upgrade navsim_envs`

### Test the installation

In order to test the install:
* create an experiment folder, we use folder `~/exp` and change into this folder.
* create a minimal yaml file with the following contents:
  ```yaml
  env_path: /path/to/Berlin_Walk_V2.x86_64
  ```
  Let us say you named this file `min_env_config.yml`. 
* Run the tests: `navsim_env_test min_env_config.yml`. If you are using `navsim`
  container then follow the instructions on the container page.

## How to use the Arora env

If you only want to use `AroraGymEnv`, then either subclass it or use it as follows:

```python
import navsim_envs
import gym

env_config = {
  "agent_car_physics": 0,
  "debug": False,
  "episode_max_steps": 1000,
  "env_gpu_id": 0,
  "env_path": "/path/to/Berlin_Walk_V2.x86_64",
  "goal": 0,
  "goal_distance": 50,
  "log_folder": "./env_log",
  "obs_mode": 0,
  "obs_height": 256,
  "obs_width": 256,
  "seed": 123,
  "start_from_episode": 1,
  "reward_for_goal": 50,
  "reward_for_no_viable_path": -50,
  "reward_step_mul": 0.1,
  "reward_collision_mul": 4,
  "reward_spl_delta_mul": 1,
  "save_actions": True,
  "save_vector_obs": True,
  "save_visual_obs": True,
  "show_visual": False,
  "task": 0,
  "timeout": 600,
  "traffic_vehicles": 0,
}

# Either use gym.make to create an env
env = gym.make("arora-v0", env_config=env_config)
# or use the AroraGymEnv constructor to create an env
env = navsim_envs.env.AroraGymEnv(env_config)
```

If you want to use our `navsim` conda environment or `navsim` container then
follow the instructions <insert_link_here>.

## Config Parameters

TODO: Explain all the above parameters here from config dictionary

### Observation Mode
* `0` - Vector - Returns \[Agent Position (3-x,y,z) ,Agent Velocity (3-x,y,z), 
  Agent Rotation(4-x,y,z,w), Goal Position (3-x,y,z,w)]  
* `1` - Visual- Returns \[\[Raw Agent Camera]\(84,84,3), 
  \[Depth Agent Camera]\(84,84,1), \[Segmentation Agent Camera]\(84,84,3)]  
* `2` - VectorVisual - Returns \[\[Raw Agent Camera]\(84,84,3), 
  \[Depth Agent Camera]\(84,84,1), \[Segmentation Agent Camera]\(84,84,3), 
  \[Agent Position (3-x,y,z), Agent Velocity (3-x,y,z), 
  Agent Rotation (4-x,y,z,w), Goal Position (3-x,y,z,w)]]  

### Segmentation Mode
* `0` - Object Seg: Each gameobject in the scene is a unique color  
* `1` - Tag Seg:  Gameobject colors are based on the tag assigned such that all 
objects with the same tag share a color. (E.g. Car, Tree, Buildings)  
* `2` - Layer Seg: Similar to tag segmentation but with the physics layers. 
Current layers (Default, Trees, Tree Colliders, Agent Vehicle, 
Autonomous Vehicle, Parked Vehicle)  

### Task
* `0` - PointNav - Agent is randomly placed along with a randomly place goal 
position. The agent must navigate to the goal position.  
* `1` - SimpleObjectNav1 - The Agent is place at a specified starting location 
(manually identified traffic intersection). Goal is a sedan 40m forward in a 
straight line of the agent. The goal is to reach that sedan.  
* `2` - ObjectNav - The Agent is randomly place and goal object is defined by 
the goal parameter. The agent must reach one instance of the goal object. 
E.g. The goal object is a sedan and there any multiple sedans in the scene. 
Reaching any of the sedans results in a success.  

### Goal : Only relevant for SimpleObjectNav and ObjectNav
* `0` - Tocus  
* `1` - sedan1  
* `2` - Car1  
* `3` - Car2  
* `4` - City Bus  
* `5` - Sporty_Hatchback  
* `Else` - SEDAN  

### Rewards
#### Reward Values
* `reward_for_goal` : For pointnav, the goal is the target position to complete the 
task.  
* `reward_for_ep` : Exploration points are randomly placed in the environment to 
reward exploration.  
* `reward_collision_mul` : This reward multiple is used to determine the reward upon collision are anything that is not a goal point or 
exploration point. This includes other cars, building, trees, etc.  
* `reward_for_no_viable_path` : The map is a tiled specified bounded by the values of env.unity_map_dims(). If the agent 
goes outside of this area and falls -15m below the environment area or enters an area outside of the navigable area then this reward is 
activated. This will also result in a reset.  
* `reward_step_mul` : This reward multiplier is used to determine the rewards given at every step in addition to any 
other reward recieved at the same step.  
* `reward_spl_delta_mul` : This reward multiplier is used to determine the reward as the agent reduces the current SPL to the goal

#### Reward Specifications
```
 
spl_delta = spl_prev_step – spl_current_step 
#If spl_delta is positive: the agent is closer to the goal according to spl 
#If spl_delta is negative: the agent is further away from the goal according to spl 

spl_reward = -(1 * reward_spl_delta_mul)  if delta==0 else spl_delta * reward_spl_delta_mul 
step_reward = -(reward_for_goal / start_spl)  * reward_step_mul 
collision_reward = reward_collision_mul * step_reward

if `agent reached goal`:
    total_reward = goal_reward 

elif `agent has no viable path`: 
    total reward = -no_viable_path_reward

else: 
    total_reward = spl_reward + step_reward + collision_reward 

```

### Agent Car Physics
* `0` - Simple : Collisions and gravity only - An agent that moves by a 
  specific distance and direction scaled by the provided action. This agent only experiences collision and gravity forces  
* `1` - Intermediate 1 : Addition of wheel torque   
* `2` - Intermediate 2 : Addition of suspension, downforce, and sideslip  
* `10` - Complex : Addition of traction control and varying surface friction  

## Action Space
    [Throttle, Steering, Brake]  
* `Throttle` : -1.0 to 1.0 : Moves the agent backward or forward  
* `Steering` : -1.0 to 1.0 : Turns the steering column of the vehicle towards 
  left or right    
* `Brake` : -1.0 to 1.0 : Reduces the agents current velocity    

### Car motion explanation based on action space

#### Simple Car Physics (Agent car physics `0`)   
In this mode, the steering and travel of a car is imitated without driven 
wheels. This means that the car will have a turning radius, but there is no 
momentum or acceleration that is experienced from torque being applied to 
wheels as in a real car.    

* `[0, 0, 0]` - No throttle, steering, or braking is applied. No agent travel.  
* `[1, 0, 0]` - Full forward throttle is applied. The agent travels forward at 
  max velocity for the duration of the step.    
* `[0, -1, 0]` - No throttle or braking is applied. Steering is applied as a 
  full left-turn, but because the forward/backward speed is zero, 
there is no travel.    
* `[1, -1, 0]` - Full forward throttle and full left-turn steering are applied. 
  The agent travels forward at a leftward angle that is equal to a fraction of 
  the max steering angle (25 degrees for the default car). This fraction is 
  dependent on the length of the step in real time.    
* `[-1, -1, 0]` - Full backward throttle and full left-turn steering are 
  applied. Similar to previous example, but with backward travel.  
* `[0.5, 0.5, 0]` - Half forward throttle and half right-turn steering are 
  applied. The agent travels forward at half its max velocity and at a lesser 
  rightward angle.  
* `[1, 0, 1]` - Full forward throttle and full braking are applied. These 
  cancel each other out and result in no agent travel.    
* `[1, 0, 0.5]` - Full forward throttle and half braking are applied. The agent 
  travels forward at half throttle.    
* `[0, 0, 1]` - Full braking is applied, with no throttle or steering. No agent 
  travel.   
    
#### Torque-Driven Car Physics (Agent car physics `>0`)  
The agent car is driven forward by applying torque to each drive wheel. The 
agent will have momentum, so travel is possible in a step where no throttle is 
input. With those differences in mind, the action space examples are similar 
with some minor behavioral differences:    

* `[0, 0, 1]` - Full braking is applied. The agent will slow to a complete stop 
  if in motion.   
* `[0, 0, 0.5]` - Half braking is applied. The agent will slow at a lesser rate 
  to the previous example, until completely stopped.   
* `[1, 0, 0]` - Full forward throttle is applied. The agent will travel forward 
  at an acceleration resulting from max wheel torque (not velocity, as in the 
  simple car physics)    
* `[1, 0, 1]` - Full forward throttle and full braking are applied. The agent 
  will not travel forward if it does not have any forward momentum, otherwise 
  the agent will slow to a complete stop.   


## Observation Space

### The vector observation space
    [Agent_Position.x, Agent_Position.y, Agent_Position.z,
    Agent_Velocity.x, Agent_Velocity.y, Agent_Velocity.z,
    Agent_Rotation.x, Agent_Rotation.y, Agent_Rotation.z, Agent_Rotation.w,
    Goal_Position.x, Goal_Position.y, Goal_Position.z,
    Proximity_Forward, Proximity_45_Left, Proximity_45_Right]

* `Proximity_*` refers to the navigable / clear space before the agent collides 
  with another object
  
### The visual observation space

    [[Raw Agent Camera],[Depth Agent Camera],[Segmentation Agent Camera]]



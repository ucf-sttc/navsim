# Navsim Environment Tutorial
NavSimGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface
The configuration provided is as follows:

## How to use the navsim env

If you only want to use `NavSimGymEnv`, then all you need to do is install
`navsim` from pip and then either subclass it or use it as follows:

```python
import navsim
import gym

env_config = navsim.util.ObjDict({
        "env_path": "/data/work/unity-envs/Build2.9.2/Berlin_Walk_V2.x86_64",
        "log_folder":"./env_log", 
        "task": 0,
        "goal": 0,
        "goal_distance": 50,
        "reward_for_goal": 50,
        "reward_for_no_viable_path":-50,
        "reward_step_mul": 0.1,
        "reward_collision_mul": 4,
        "reward_spl_delta_mul": 1,
        "agent_car_physics": 0,   
        "debug":False,
        "obs_mode":0,
        "seed":123,
        "save_vector_obs":True,
        "save_visual_obs":True
    })
    
env = gym.make("navsim-v0", env_config=env_config) 
# or use the following method to create an env
env = navsim.NavSimGymEnv(env_config)
```

If you want to use our `navsim` conda environment or `navsim` container then
follow the instructions <insert_link_here>.

## Config Parameters

```python
env_config = ObjDict({
    "log_folder": "unity.log",
    "seed": 123,
    "timeout": 600,
    "worker_id": 0,
    "base_port": 5005,
    "obs_mode": 2,
    "segmentation_mode": 1,
    "task": 0,
    "goal": 0,
    "goal_distance":50
    "max_steps": 10,
    "reward_for_goal": 50,
    "reward_for_ep": 0.005,
    "reward_for_other": -0.1,
    "reward_for_falling_off_map": -50,
    "reward_for_step": -0.0001,
    "agent_car_physics": 0,
    "episode_max_steps": 10,
    "env_path":args["env_path"]
})
```
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
* `reward_for_goal` : For pointnav goal is the target position to complete the 
task.  
* `reward_for_ep` : Exploration points are randomly placed in the environment to 
reward exploration.  
* `reward_for_other` : Other collision are anythin that is not a goal point or 
exploration point, this includes other cars, building, trees, etc.  
* `reward_for_falling_off_map` :  The map is a tiled XXkm area. If the agent 
goes outside of this area falls XXm below the environment area this reward is 
activated. This will also result in a reset.  
* `reward_for_step` : This reward will be given at every step in addition to any 
other reward recieved at the same step.  

### Agent Car Physics
* `0` - Simple : Collisions and gravity only - An agent that moves by a 
  specific distance and direction scaled by the provided action. This agent only experiences collision and gravity forces  
* `1` - Intermediate 1 : Addition of wheel torque   
* `2` - Intermediate 2 : Addition of suspension, downforce, and sideslip  
* `10` - Complex : Addition of traction control and varying surface friction  

## Action Space: 
    [Throttle, Steering, Brake]  
* `Throttle` : -1.0 to 1.0 : Moves the agent backward or forward  
* `Steering` : -1.0 to 1.0 : Turns the steering column of the vehicle towards 
  left or right    
* `Brake` : 0.0 to 1.0 : Reduces the agents current velocity    

## Car Motion Explanation:
**Simple Car Physics** (Agent car physics `0`)   
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
    
**Torque-Driven Car Physics** (Agent car physics `>0`)  
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


## Observation Space: 


### The vector observation space
    Agent_Position.x, Agent_Position.y, Agent_Position.z,
    Agent_Velocity.x, Agent_Velocity.y, Agent_Velocity.z,
    Agent_Rotation.x, Agent_Rotation.y, Agent_Rotation.z, Agent_Rotation.w,
    Goal_Position.x, Goal_Position.y, Goal_Position.z

### The visual observation space

    [[Raw Agent Camera],[Depth Agent Camera],[Segmentation Agent Camera]]

## Queries from the Env

### Map
Used to request and receive a binary navigable map. The binary map indicates 
navigable and obstacle areas. 

Map requests to Unity are sent using: 

    NavSimGymEnv.start_navigable_map(resolution_x, resolution_y, cell_occupancy_threshold)

The map is then retrieved with:

    NavSimGymEnv.get_navigable_map()

Parameter value ranges:    
```
                          Min   Max      
resolution_x              1     3276   
resolution_y              1     2662    
cell_occupancy_threshold  0     1.0
```

The raw map array received from the Unity game is a row-major 1D flattened 
bitpacked array with the y-axis data ordered for image output 
(origin at top left).    

For example, if reshaping to a 2D array without reordering with 
dimensions `(resolution_y, resolution_x)`, then the desired coordinate `(x,y)` 
is at array element `[resolution_y-1-y, x]`.    
Finding the agent map position based on world position*:    
`map_x = floor(world_x / (max_x / resolution_x) )`    
`map_y = (resolution_y - 1) - floor(world_z / (max_y / resolution_y) )`    

*Note: When converting from the 3-dimensional world position to the 
2-dimensional map, the world y-axis is omitted. The map's y-axis represents 
the world's z-axis.    

### Position Scan - Not Available
Given a position and this returns the attribution data of the first object 
found at the given position. Objects are searched for within a 1 meter radius 
of the given position. If the position is not loaded in the environment then 
None will be returned. 

### Shortest Path from Starting Location to Goal

`ShortestPath` : Returns the shortest path value from the agent's start 
location to the goal position from the navigable area.

	NavSimGymEnv.get_shortest_path_length()

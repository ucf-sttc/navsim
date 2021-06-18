#Navsim Environment
NavSimGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface
The configuration provided is as follows:


##Config Parameters
.. code-block:: python

    env_config = ObjDict({
        "log_folder": "unity.log",
        "seed": 123,
        "timeout": 600,
        "worker_id": 0,
        "base_port": 5005,
        "observation_mode": 2,
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

**Observation Mode**
`0` - Vector - Returns [Agent Position (3-x,y,z) ,Agent Velocity (3-x,y,z), Agent Rotation(4-x,y,z,w), Goal Position (3-x,y,z,w)]
`1` - Visual- Returns [[Raw Agent Camera](84,84,3), [Depth Agent Camera](84,84,1), [Segmentation Agent Camera](84,84,3)]
`2` - VectorVisual - Returns [[Raw Agent Camera](84,84,3), [Depth Agent Camera](84,84,1), [Segmentation Agent Camera](84,84,3), [Agent Position (3-x,y,z) ,Agent Velocity (3-x,y,z), Agent Rotation(4-x,y,z,w), Goal Position (3-x,y,z,w)]]

**Segmentation Mode**
`0` - Object Seg: Each gameobject in the scene is a unique color
`1` - Tag Seg:  Gameobject colors are based on the tag assigned such that all objects with the same tag share a color. (E.g. Car, Tree, Buildings)
`2` - Layer Seg: Similar to tag segmentation but with the physics layers. Current layers (Default, Trees, Tree Colliders, Agent Vehicle, Autonomous Vehicle, Parked Vehicle)

**Task**
`0` - PointNav - Agent is randomly placed along with a randomly place goal position. The agent must navigate to the goal position.
`1` - SimpleObjectNav1 - The Agent is place at a specified starting location (manually identified traffic intersection). Goal is a sedan 40m forward in a straight line of the agent. The goal is to reach that sedan.
`2` - ObjectNav - The Agent is randomly place and goal object is defined by the goal parameter. The agent must reach one instance of the goal object. E.g. The goal object is a sedan and there any multiple sedans in the scene. Reaching any of the sedans results in a success.

**Goal** : Only relevant for SimpleObjectNav and ObjectNav
`0` - Tocus
`1` - sedan1
`2` - Car1
`3` - Car2
`4` - City Bus
`5` - Sporty_Hatchback
`Else` - SEDAN

**Rewards**
`reward_for_goal` : For pointnav goal is the target position to complete the task.
`reward_for_ep` : Exploration points are randomly placed in the environment to reward exploration.
`reward_for_other` : Other collision are anythin that is not a goal point or exploration point, this includes other cars, building, trees, etc.
`reward_for_falling_off_map` :  The map is a tiled XXkm area. If the agent goes outside of this area falls XXm below the environment area this reward is activated. This will also result in a reset.
`reward_for_step` : This reward will be given at every step in addition to any other reward recieved at the same step


####Action Space: 
[Throttle, Steering, Brake]
`Throttle` : -1.0 to 1.0 : Moves the agent forward
`Steering` : -1.0 to 1.0 : Turns the steering column of the vehicle left or right
`Brake` : 0.0 to 1.0 : Reduces the agents current velocity

####Observation Space: 
[[Raw Agent Camera],[Depth Agent Camera],[Segmentation Agent Camera],[Agent Position, Agent Velocity, Agent Rotation, Goal Position]]

**The vector observation space**:
    Agent_Position.x, Agent_Position.y, Agent_Position.z,
    Agent_Velocity.x, Agent_Velocity.y, Agent_Velocity.z,
    Agent_Rotation.x, Agent_Rotation.y, Agent_Rotation.z, Agent_Rotation.w,
    Goal_Position.x, Goal_Position.y, Goal_Position.z


engine_side_channel = EngineConfigurationChannel()
engine_side_channel.set_configuration_parameters(time_scale=2, quality_level = 0)

##Custom Side Channels
Side channels are used to set and recieve parameters to and from the Unity game

###Map
Used to request and recieve a binary navigable map. The binary map indicates navigable and obstacle areas.
Map requests to Unity are sent using: 

	NavSimGymEnv.start_navigable_map(resolution_x, resolution_y, cell_occupancy_threshold)

The map is then retrieved with:

	 NavSimGymEnv.get_navigable_map()

###Postion Scan - Not Available
Given a position and this returns the attribution data of the first object found at the given position. Objects are searched for within a 1 meter radious of the given position. If the position is not loaded in the environmen then None will be returned. 

###Float
Used to return properties that are float values.

`ShortestPath` : Returns the shortest path value from the agent's start location to the goal position from the navigable area.

	NavSimGymEnv.get_shortest_path_length()

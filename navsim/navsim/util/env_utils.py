def env_info(env):
    """Prints the information about the environment

    """
    print('-----------')
    print("Env Info")
    print('-----------')
    if env.spec is not None:
        print(env.spec.id)
    print('Action Space:', env.action_space)
    # TODO: print env_config if it exists
    # print('Observation Mode:', env.obs_mode)
    # print('Gym Observation Space:', self.genv.observation_space)
    # print('Gym Observation Space Shape:', self.genv.observation_space.shape)
    print('Observation Space:', env.observation_space)
    if hasattr(env.observation_space, 'spaces'):
        print('Observation Space Spaces:',
              [obs for obs in env.observation_space.spaces])
    #    print('Observation Space Types:', [type(obs) for obs in env.observation_space.spaces])
    print('Reward Range:', env.reward_range)
    print('Metadata:', env.metadata)
    print('--------------------------------------')
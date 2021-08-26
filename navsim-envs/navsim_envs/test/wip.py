class TestAroraGymEnv_TimeCheck:
    """
    For each observation mode: -- class-wide obs_mode param
        - test binary init time - create env again and again within the test
        - test reset time - use the class env
        - test step time - use the class env
    """

    def test_env_init_time(self, request, env_config_4_session):
        logger.info(f"=========== Running {request.node.name}")

        # TODO: Shouldnt we check that each connection time should be <30 and not on average ?
        # TODO: Shouldnt we check for reset times also ?
        # TODO: Shouldnt we combine all the time test checks into one test ?
        init_times = []
        trials = 5
        for i in range(trials):
            start_time = time.time()
            env = gym.make(env_id, env_config=env_config_4_session)
            end_time = time.time()
            init_times.append(end_time - start_time)
            env.close()
            del env

        logger.info(f'{init_times}')
        assert sum(init_times) / len(init_times) < 30

    def test_reset_and_step_time(self, request, env_4_class, env_config_4_session):
        logger.info(f"=========== Running {request.node.name}")

        env = env_4_class(env_config_4_session)
        reset_times = []
        avg_step_times = []
        trials = 5
        for i in range(trials):
            step_deltas = []
            done = False
            start_time = time.time()
            _ = env.reset()
            end_time = time.time()
            reset_times.append(end_time - start_time)
            while not done:
                start_time = time.time()
                o, r, done, i = env.step([1, 0, -1])
                end_time = time.time()

                step_deltas.append(end_time - start_time)
            avg_step_times.append(sum(step_deltas) / len(step_deltas))

        logger.info(f'reset_times: {reset_times}')
        logger.info(f'avg_step_times: {avg_step_times}')

    def test_step_time(self, request, env_4_class, env_config_4_session):
        logger.info(f"=========== Running {request.node.name}")

        env = gym.make(env_id, env_config=env_config_4_session)
        episode_avg_step_times = []
        trials = 5
        for i in range(0, trials):
            step_deltas = []
            done = False
            env.reset()
            while not done:
                start_time = time.time()
                o, r, done, i = env.step([1, 0, -1])
                end_time = time.time()

                step_deltas.append(end_time - start_time)
            episode_avg_step_times.append(sum(step_deltas) / len(step_deltas))

        env.close()
        del env
        logger.info(f'{episode_avg_step_times}')
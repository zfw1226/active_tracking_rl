from gym.envs.registration import register

for maze in ['Random', 'U', 'Block', 'Empty']:
    for obs in ['Full', 'Partial']:
        for target in ['Adv', 'PZR', 'Nav', 'Ram']:
                for level in range(2):
                    entry_point = 'gym_track2d.envs:Track1v1Env'
                    register(
                        id='Track2D-{maze}{obs}{target}-v{level}'.format(
                            maze=maze, obs=obs, level=level, target=target),
                        entry_point=entry_point,
                        kwargs={'maze_type': maze,
                                'obs_type': obs,
                                'level': level,
                                'target_mode': target,
                                },
                        max_episode_steps=500,
                    )




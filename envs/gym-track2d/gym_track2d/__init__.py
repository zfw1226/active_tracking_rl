from gym.envs.registration import register

for map_name in ['Maze', 'Block', 'Empty']:
    for obs in ['Full', 'Partial']:
        for target in ['Adv', 'PZR', 'Far', 'Nav', 'Ram', 'RPF']:
                for level in range(2):
                    entry_point = 'gym_track2d.envs:Track1v1Env'
                    register(
                        id='Track2D-{map_name}{obs}{target}-v{level}'.format(
                            map_name=map_name, obs=obs, level=level, target=target),
                        entry_point=entry_point,
                        kwargs={'map_type': map_name,
                                'obs_type': obs,
                                'level': level,
                                'target_mode': target,
                                },
                        max_episode_steps=500,
                    )




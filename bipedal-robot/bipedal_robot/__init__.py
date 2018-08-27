from gym.envs.registration import register

register(
    id='bipedalRobot-v0',
    entry_point='bipedal_robot.envs:BipedalRobotEnv',
)

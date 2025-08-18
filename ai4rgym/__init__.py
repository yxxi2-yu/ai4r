from gymnasium.envs.registration import register

register(
    id="ai4rgym/autonomous_driving_env",
    entry_point="ai4rgym.envs:AutonomousDrivingEnv",
    max_episode_steps=None,
    reward_threshold=None,
    nondeterministic=False,
    order_enforce=True,
)

# register(
#     id="ai4rgym/pendulum",
#     entry_point="ai4rgym.envs:PendulumEnv",
#     max_episode_steps=None,
#     reward_threshold=None,
#     nondeterministic=False,
#     order_enforce=True,
# )

# register(
#    id="ai4rgym/rocket2d",
#    entry_point="ai4rgym.envs:RocketEnv",
#    max_episode_steps=None,
#    reward_threshold=None,
#    nondeterministic=False,
#    order_enforce=True,
# )

# register(
#     id="ai4rgym/service_bot",
#     entry_point="ai4rgym.envs:ServiceBot",
#     max_episode_steps=None,
#     reward_threshold=None,
#     nondeterministic=False,
#     order_enforce=True,
# )

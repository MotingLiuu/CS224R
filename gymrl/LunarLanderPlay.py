from gymnasium.utils.play import play
import gymnasium as gym

play(gym.make('LunarLander-v3', render_mode='rgb_array'),
    keys_to_action={'w': 2, 'a': 1, 'd': 3}, noop=0)
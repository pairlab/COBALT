import numpy as np
from robosuite.wrappers import Wrapper

from utils.custom_venv import DummyVectorEnv, SubprocVectorEnv


class EnvStateWrapper(Wrapper):
    """
    Wrapper to capture full state information for reset including sim, controller, and gripper states
    """

    def __init__(self, env):
        super(EnvStateWrapper, self).__init__(env)

    def flip_image(self, obs):
        for k in obs.keys():
            if (
                k.endswith("_image")
                or k.endswith("_depth")
                or k.endswith("_segmentation_instance")
            ):
                obs[k] = obs[k][::-1, :, :]
        return obs

    def get_env_state(self):
        obs = self.flip_image(self.env._get_observations())
        obs["sim_state"] = np.array(self.env.sim.get_state().flatten())
        for idx in range(len(self.env.robots)):
            obs[f"robot{idx}_base_ori"] = self.env.robots[idx].base_ori
            obs[f"robot{idx}_hand_orn"] = self.env.robots[idx]._hand_orn

        return obs

    def set_env_state(self, env_state):
        self.env.set_state(env_state["sim_state"])
        self.env.sim.forward()
        self._set_gripper_state(env_state["gripper_state"])


class ResetWrapper(Wrapper):
    """
    Wrapper to reset the environment.
    """

    def __init__(self, env):
        super(ResetWrapper, self).__init__(env)

    def reset(self):
        self.env.reset()

        # dummy actions all zeros for initial physics simulation
        dummy = np.zeros(self.action_dim)
        dummy[-1] = -1.0  # set the last action to -1 to open the gripper
        for _ in range(5):
            obs, _, _, _ = self.env.step(dummy)

        return obs


class SuccessWrapper(Wrapper):
    """Wrapper to check for success in the environment"""

    def __init__(self, env):
        super(SuccessWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = self.env._check_success()
        return obs, reward, done, info


def merge_dict(dict_obj):
    merged_dict = {}
    for k in dict_obj[0].keys():
        merged_dict[k] = np.stack([d[k] for d in dict_obj], axis=0)
    return merged_dict


######################################
### Custom Vectorized environments ###
######################################


class StackDummyVectorEnv(DummyVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, id=None):
        obs = super().reset(id=id)
        return merge_dict(obs)

    def step(self, action: np.ndarray, id=None):
        obs, reward, done, info = super().step(action, id)
        return merge_dict(obs), reward, done, merge_dict(info)

    def regenerate_obs_from_state(self, mujoco_state):
        obs = super().regenerate_obs_from_state(mujoco_state)
        return merge_dict(obs)


class StackSubprocVectorEnv(SubprocVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        return merge_dict(obs)

    def reset_envs(self, env_ids):
        obs = super().reset(id=env_ids)
        return merge_dict(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return merge_dict(obs), reward, done, merge_dict(info)

    def regenerate_obs_from_state(self, mujoco_state):
        obs = super().regenerate_obs_from_state(mujoco_state)
        return merge_dict(obs)

    def get_env_state(self):
        obs = super().get_env_state()
        return merge_dict(obs)

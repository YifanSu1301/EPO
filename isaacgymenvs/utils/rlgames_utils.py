# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import os
from collections import deque
from typing import Callable, Dict, Tuple, Any

import os
import gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from isaacgymenvs.pbt.pbt import PbtAlgoObserver

from isaacgymenvs.tasks import isaacgym_task_map
from rl_games.common.custom_utils import remove_envs_from_info
from isaacgymenvs.utils.utils import set_seed, flatten_dict

from omegaconf import DictConfig

from scipy.stats import skew

def multi_gpu_get_rank(multi_gpu):
    if multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        print("GPU rank: ", rank)
        return rank

    return 0


def get_rlgames_env_creator(
        # used to create the vec task
        seed: int,
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        # used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
        virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
        force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """
        if multi_gpu:

            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))

            # local rank of the GPU in a node
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            world_size = int(os.getenv("WORLD_SIZE", "1"))

            print(f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}")

            _sim_device = f'cuda:0'
            _rl_device = f'cuda:0'

            task_config['rank'] = local_rank
            task_config['rl_device'] = _rl_device
        else:
            _sim_device = sim_device
            _rl_device = rl_device

        # create native task and pass custom config
        env = isaacgym_task_map[task_name](
            cfg=task_config,
            rl_device=_rl_device,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env

_UNINITIALIZED_VALUE = float(-1e9)

class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None

        self.ep_infos = []
        self.direct_info = {}

        self.episode_cumulative = dict()
        self.episode_cumulative_avg = dict()
        self.new_finished_episodes = False

    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices, **kwargs):
        assert isinstance(infos, dict), 'RLGPUAlgoObserver expects dict info'
        if not isinstance(infos, dict):
            return

        if kwargs.get("ignore_env_boundary", 0) > 0:
            infos = remove_envs_from_info(copy.deepcopy(infos), kwargs["ignore_env_boundary"])
            done_indices = done_indices[done_indices >= kwargs["ignore_env_boundary"]].unsqueeze(-1) - kwargs["ignore_env_boundary"]

        if 'episode' in infos:
            self.ep_infos.append(infos['episode'])

        if 'episode_cumulative' in infos:
            for key, value in infos['episode_cumulative'].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value

            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()

                for key, value in infos['episode_cumulative'].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque([], maxlen=self.algo.games_to_track)

                    self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
                    self.episode_cumulative[key][done_idx] = 0

        # turn nested infos into summary keys (i.e. infos['scalars']['lr'] -> infos['scalars/lr']
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            infos_flat = flatten_dict(infos, prefix='', separator='/')
            self.direct_info = {}
            for k, v in infos_flat.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v
        
        for tag in ['successes', 'closest_keypoint_max_dist', 'discounted_reward']:
            if tag in infos:
                self.direct_info[tag] = infos[tag].mean()
                self.direct_info[f'{tag}_median'] = torch.median(infos[tag]).item()
                self.direct_info[f'{tag}_max'] = infos[tag].max()
                for key in infos:
                    if key.startswith(f'{tag}_per_block'):
                        self.direct_info[key] = torch.mean(infos[key]).item()
        
        if 'true_objective' in infos:
            self.direct_info['true_objective_mean'] = infos['true_objective'].mean()
            self.direct_info['true_objective_max'] = infos['true_objective'].max()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, frame)
            self.ep_infos.clear()
        
        # log these if and only if we have new finished episodes
        if self.new_finished_episodes:
            for key in self.episode_cumulative_avg:
                self.writer.add_scalar(f'episode_cumulative/{key}', np.mean(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_min/{key}_min', np.min(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_max/{key}_max', np.max(self.episode_cumulative_avg[key]), frame)
            self.new_finished_episodes = False
        
        print('Frame: ', frame, end=' ')
        for k, v in self.direct_info.items():
            print(f'{k}/frame', v, end=' ')
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, frame)
            self.writer.add_scalar(f'{k}/time', v, frame)
        print()


class EPOObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.algo = None
        self.writer = None

        self.ep_infos = []
        self.direct_info = {}

        self.episode_cumulative = dict()
        self.episode_cumulative_avg = dict()
        self.new_finished_episodes = False

        # EPO
        self.best_embeddings = True
        self.num_agent = cfg.task.env.numEnvs // cfg.train.params.config.expl_coef_block_size
        self.sapg_iteration = -1

        self.finished_agents = set()
        self.last_target_objectives = None
        self.best_objective_curr_iteration: Optional[float] = None

        # EPO params (fixed for now)
        self.interval_steps = 20000000
        self.warm_up_steps = 200000000


    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer
    
    def after_steps(self):
        '''Called after each step of the algorithm.
        Eliminate worse embeddings and replace them with merged better ones.
        '''
        # only need when using sapg plus
        if not self.best_embeddings:
            return
                
        print('After steps (# frames):', self.algo.frame, self.algo.device)
        if self.sapg_iteration == -1:
            self.sapg_iteration = self.algo.frame // self.interval_steps
            self.init_env_frames = self.algo.frame
            print('SAPG+ Init:', self.sapg_iteration)
        
        env_frames: int = self.algo.frame
        iteration = env_frames // self.interval_steps

        if iteration <= self.sapg_iteration:
            return
        
        if not self.target_objective_known:
            # not enough data yet to calcuate avg true_objective
            print(
                f" Not enough episodes finished, wait for more data..."
            )
            return
        
        # warm-up steps
        if env_frames < self.warm_up_steps:
            print(f"Giving this policy more time to adjust to the latest parameters... ")
            return


        policies = list(range(self.num_agent))
        best_objective_curr_iteration_per_block = self.best_objective_curr_iteration_per_block
        policies_sorted = sorted(zip(best_objective_curr_iteration_per_block, policies), reverse=True)

        objectives = [obj for obj, _ in policies_sorted]
        policies = [pol for _, pol in policies_sorted]
        best_policies = policies[:2]
        worst_policy = policies[-1]

        print(f"Best_policies={best_policies}, worst_policies={worst_policy}: Start replacing embeddings")
        objectives_filtered = [obj for obj in objectives if obj > _UNINITIALIZED_VALUE]
        objective_delta = objectives_filtered[0] - objectives_filtered[-1]
        objective_std = np.std(objectives_filtered)
        # get the skewness of the objectives
        objective_skewness = skew(objectives_filtered)

        self.writer.add_scalar('auxiliary/objective_std', objective_std, env_frames)
        self.writer.add_scalar('auxiliary/objective_delta', objective_delta, env_frames)
        self.writer.add_scalar('auxiliary/objective_skewness', objective_skewness, env_frames)
     
        # Eliminate the worst policy
        for i in range(1):
            worst_policy = policies[-1-i]
            self.algo.model.a2c_network.merge_extra_params(best_policies, worst_policy)

        # reset best_objective_curr_iteration_per_block
        self.best_objective_curr_iteration_per_block = [_UNINITIALIZED_VALUE] *self.num_agent
        self.target_objective_known = False
        self.sapg_iteration = iteration

        return True
    def preprocess_image(self, image):
        # If the image is 1D or has an invalid shape, reshape it
        if image.ndim == 3 and image.shape[2] == 1:  # If single-channel with shape (H, W, 1)
            image = np.squeeze(image, axis=-1)  # Remove the channel dimension

        elif image.ndim == 1 or image.shape == (1, 1, 480):  # Handle unusual shapes
            image = image.reshape((int(np.sqrt(image.size)), -1))  # Reshape to a square if possible

        # Normalize or cast to uint8 if required
        if image.dtype != np.uint8:
            image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
        
        # Ensure RGB format (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
        
        return image

        
    def process_infos(self, infos, done_indices, **kwargs):
        assert isinstance(infos, dict), 'RLGPUAlgoObserver expects dict info'
        if not isinstance(infos, dict):
            return
        
        
        
        if self.best_embeddings and self.last_target_objectives == None:
            # init
            print("Initializing last_target_objectives")
            # all keys in infos
            print("all keys in infos: ", infos.keys())
            self.last_target_objectives = [_UNINITIALIZED_VALUE] * (infos['true_objective'].shape[0] - kwargs.get("ignore_env_boundary", 0))
            # breakpoint()
            self.curr_target_objective_value_per_block = [_UNINITIALIZED_VALUE] *self.num_agent
            self.best_objective_curr_iteration_per_block = [_UNINITIALIZED_VALUE] * self.num_agent

        if kwargs.get("ignore_env_boundary", 0) > 0:
            infos = remove_envs_from_info(copy.deepcopy(infos), kwargs["ignore_env_boundary"])
            done_indices = done_indices[done_indices >= kwargs["ignore_env_boundary"]].unsqueeze(-1) - kwargs["ignore_env_boundary"]
            

        if 'episode' in infos:
            self.ep_infos.append(infos['episode'])

        if 'episode_cumulative' in infos:
            for key, value in infos['episode_cumulative'].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value

            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()

                for key, value in infos['episode_cumulative'].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque([], maxlen=self.algo.games_to_track)

                    self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
                    self.episode_cumulative[key][done_idx] = 0

        # turn nested infos into summary keys (i.e. infos['scalars']['lr'] -> infos['scalars/lr']
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            infos_flat = flatten_dict(infos, prefix='', separator='/')
            self.direct_info = {}
            for k, v in infos_flat.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v
        
        for tag in ['successes', 'closest_keypoint_max_dist', 'discounted_reward', 'true_objective']:
            if tag in infos:
                self.direct_info[tag] = infos[tag].mean()
                self.direct_info[f'{tag}_median'] = torch.median(infos[tag]).item()
                self.direct_info[f'{tag}_max'] = infos[tag].max()
                for key in infos:
                    if key.startswith(f'{tag}_per_block'):
                        self.direct_info[key] = torch.mean(infos[key]).item()
        
        if 'true_objective' in infos:
            self.direct_info['true_objective_mean'] = infos['true_objective'].mean()
            self.direct_info['true_objective_max'] = infos['true_objective'].max()
            if self.best_embeddings:
                done_indices_lst = done_indices.squeeze(-1).tolist()
                self.finished_agents.update(done_indices_lst)
                # print("done indices list: ", done_indices_lst)

                for done_idx in done_indices_lst:
                    true_objective_value = infos["true_objective"][done_idx].item()
                    self.last_target_objectives[done_idx] = true_objective_value

                # # last result for all episodes
                b_size = infos['true_objective'].shape[0] // self.num_agent
                self.target_objective_known = len(self.finished_agents) >= len(self.last_target_objectives)
                # print("!!!!!! known: ", self.target_objective_known, self.finished_agents, len(self.last_target_objectives))
                if self.target_objective_known:
                # print("update true objective")
                # breakpoint()
                    self.curr_target_objective_value = float(np.mean(self.last_target_objectives))
                    for i in range(self.num_agent):
                        self.curr_target_objective_value_per_block[i] = float(np.mean(self.last_target_objectives[i*b_size:(i+1)*b_size]))
        else:
            # environment does not specify "true objective", use regular reward
            # in this case, be careful not to include reward shaping coefficients into the mutation config
            if self.best_embeddings:
                self.target_objective_known = self.algo.game_rewards.current_size >= self.algo.games_to_track
                if self.target_objective_known:
                    self.curr_target_objective_value = float(self.algo.mean_rewards)

        if self.best_embeddings and self.target_objective_known:
            for i in range(self.num_agent):
                if self.curr_target_objective_value_per_block[i] > self.best_objective_curr_iteration_per_block[i]:
                    print(
                        f"New best objective value {self.curr_target_objective_value_per_block[i]} in iteration {self.sapg_iteration} block {i}"
                    )
                    self.best_objective_curr_iteration_per_block[i] = self.curr_target_objective_value_per_block[i]


    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, frame)
            self.ep_infos.clear()
        
        # log these if and only if we have new finished episodes
        if self.new_finished_episodes:
            for key in self.episode_cumulative_avg:
                self.writer.add_scalar(f'episode_cumulative/{key}', np.mean(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_min/{key}_min', np.min(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_max/{key}_max', np.max(self.episode_cumulative_avg[key]), frame)
            self.new_finished_episodes = False
        
        print('Frame: ', frame, end=' ')
        for k, v in self.direct_info.items():
            print(f'{k}/frame', v, end=' ')
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, frame)
            self.writer.add_scalar(f'{k}/time', v, frame)
        print()


class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        ret_val = None
        for o in self.observers:
            if isinstance(o, PbtAlgoObserver):
                ret_val = getattr(o, method)(*args_, **kwargs_)
            else:
                getattr(o, method)(*args_, **kwargs_)
        return ret_val

    def before_init(self, base_name, config, experiment_name):
        self._call_multi('before_init', base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi('after_init', algo)

    def process_infos(self, infos, done_indices, **kwargs):
        self._call_multi('process_infos', infos, done_indices, **kwargs)

    def after_steps(self):
        return self._call_multi('after_steps')

    def after_clear_stats(self):
        self._call_multi('after_clear_stats')

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi('after_print_stats', frame, epoch_num, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, 'set_train_info'):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        if hasattr(self.env, 'get_env_state'):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        if hasattr(self.env, 'set_env_state'):
            self.env.set_env_state(env_state)
    
    def change_on_restart(self, task_cfg):
        if hasattr(self.env, 'change_on_restart'):
            self.env.change_on_restart(task_cfg)


class ComplexObsRLGPUEnv(vecenv.IVecEnv):
    
    def __init__(
        self,
        config_name,
        num_actors,
        obs_spec: Dict[str, Dict],
        **kwargs,
    ):
        """RLGPU wrapper for Isaac Gym tasks.

        Args:
            config_name: Name of rl games env_configurations configuration to use.
            obs_spec: Dictinoary listing out specification for observations to use.
                eg.
                {
                 'obs': {'names': ['obs_1', 'obs_2'], 'concat': True, space_name: 'observation_space'},},
                 'states': {'names': ['state_1', 'state_2'], 'concat': False, space_name: 'state_space'},}
                }
                Within each, if 'concat' is set, concatenates all the given observaitons into a single tensor of dim (num_envs, sum(num_obs)).
                    Assumes that each indivdual observation is single dimensional (ie (num_envs, k), so image observation isn't supported).
                    Currently applies to student and teacher both.
                "space_name" is given into the env info which RL Games reads to find the space shape
        """
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        self.obs_spec = obs_spec

    def _generate_obs(
        self, env_obs: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate the RL Games observations given the observations from the environment.

        Args:
            env_obs: environment observations
        Returns:
            Dict which contains keys with values corresponding to observations.
        """
        # rl games expects a dictionary with 'obs' and 'states'
        # corresponding to the policy observations and possible asymmetric
        # observations respectively

        rlgames_obs = {k: self.gen_obs_dict(env_obs, v['names'], v['concat']) for k, v in self.obs_spec.items()}

        return rlgames_obs

    def step(
        self, action: torch.Tensor
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Step the Isaac Gym task.

        Args:
            action: Enivronment action.
        Returns:
            observations, rewards, dones, infos
            Returned obeservations are a dict which contains key 'obs' corresponding to a dictionary of observations,
            and possible 'states' key corresponding to dictionary of privileged observations.
        """
        env_obs, rewards, dones, infos = self.env.step(action)
        rlgames_obs = self._generate_obs(env_obs)
        return rlgames_obs, rewards, dones, infos

    def reset(self) -> Dict[str, Dict[str, torch.Tensor]]:
        env_obs = self.env.reset()
        return self._generate_obs(env_obs)

    def get_number_of_agents(self) -> int:
        return self.env.get_number_of_agents()

    def get_env_info(self) -> Dict[str, gym.spaces.Space]:
        """Gets information on the environment's observation, action, and privileged observation (states) spaces."""
        info = {}
        info["action_space"] = self.env.action_space

        for k, v in self.obs_spec.items():
            info[v['space_name']] = self.gen_obs_space(v['names'], v['concat'])

        return info
    
    def gen_obs_dict(self, obs_dict, obs_names, concat):
        """Generate the RL Games observations given the observations from the environment."""
        if concat:
            return torch.cat([obs_dict[name] for name in obs_names], dim=1)
        else:
            return {k: obs_dict[k] for k in obs_names}
            

    def gen_obs_space(self, obs_names, concat):
        """Generate the RL Games observation space given the observations from the environment."""
        if concat:
            return gym.spaces.Box(
                low=-np.Inf,
                high=np.Inf,
                shape=(sum([self.env.observation_space[s].shape[0] for s in obs_names]),),
                dtype=np.float32,
            )
        else:        
            return gym.spaces.Dict(
                    {k: self.env.observation_space[k] for k in obs_names}
                )

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, 'set_train_info'):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        if hasattr(self.env, 'get_env_state'):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        if hasattr(self.env, 'set_env_state'):
            self.env.set_env_state(env_state)                

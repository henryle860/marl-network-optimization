import os
import time
import csv
import numpy as np
import random
from typing import Dict, List, Tuple
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import argparse

# Different routing scenarios providing clear signals to root interfaces
ROUTING_SCENARIO = 3
# Seeding for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Constants
# NOTE: kept at 0.0 for training (no incidents during learning). For testing/evaluation
# runs with incidents, this was manually raised before running (e.g. to 0.1-0.3) and
# reset to 0.0 afterwards. Not exposed as a CLI arg yet.
INCIDENT_PROBABILITY = 0.0
INCIDENT_DATARATE = 10.0
BUFFER_SIZE = 10000
DEFAULT_DATARATE = 50.0
MAX_DATARATE = 100.0
MIN_DATARATE = 10.0
STEP_SIZE = 5
ROOT_OBS_DIM = 10
LEAF_OBS_DIM = 4
CSV_FILE_TESTING = 'Training_Results/ExpandedTreeTopology/MARL/MARL_ExpandedTreeTopology_Simulation_Testing.csv'

# Training traffic patterns
TRAFFIC_PROFILES = {
    'decreaseOne': 30.0,
    'increaseTwo': 70.0,
    'decreaseTwo': 20.0,
    'increaseOne': 80.0,

}


# Set at runtime via set_routing_scenario()
def set_routing_scenario(scenario: int):
    global ROUTING_SCENARIO
    if scenario not in [0, 1, 2, 3]:
        raise ValueError('Invalid scenario')
    ROUTING_SCENARIO = scenario
    print(f'*** Routing scenario set to {ROUTING_SCENARIO}')


class CSVLogger:
    """Buffered CSV logger for network simulation data."""

    # Fixed headers that matches the 10-switch tree topology
    HEADERS = [
        # Base metrics
        'episode_count', 'step_count', 'traffic_pattern', 'total_reward', 'incident_interfaces',

        # s1 (root switch - 3 interfaces)
        's1-eth2_datarate', 's1-eth2_throughput', 's1-eth2_packetloss', 's1-eth2_action',
        's1-eth3_datarate', 's1-eth3_throughput', 's1-eth3_packetloss', 's1-eth3_action',
        's1-eth4_datarate', 's1-eth4_throughput', 's1-eth4_packetloss', 's1-eth4_action',
        's1_reward',

        # s2 (leaf switch - 1 interface)
        's2-eth2_datarate', 's2-eth2_throughput', 's2-eth2_packetloss', 's2-eth2_action',
        's2_reward',

        # s3 (leaf switch - 1 interface)
        's3-eth2_datarate', 's3-eth2_throughput', 's3-eth2_packetloss', 's3-eth2_action',
        's3_reward',

        # s4 (leaf switch - 1 interface)
        's4-eth2_datarate', 's4-eth2_throughput', 's4-eth2_packetloss', 's4-eth2_action',
        's4_reward',

        # s5 (leaf switch - 1 interface)
        's5-eth2_datarate', 's5-eth2_throughput', 's5-eth2_packetloss', 's5-eth2_action',
        's5_reward',

        # s6 (leaf switch - 1 interface)
        's6-eth2_datarate', 's6-eth2_throughput', 's6-eth2_packetloss', 's6-eth2_action',
        's6_reward',

        # s7 (leaf switch - 1 interface)
        's7-eth2_datarate', 's7-eth2_throughput', 's7-eth2_packetloss', 's7-eth2_action',
        's7_reward',

        # s8 (leaf switch - 1 interface)
        's8-eth2_datarate', 's8-eth2_throughput', 's8-eth2_packetloss', 's8-eth2_action',
        's8_reward',

        # s9 (leaf switch - 1 interface)
        's9-eth2_datarate', 's9-eth2_throughput', 's9-eth2_packetloss', 's9-eth2_action',
        's9_reward',

        # s10 (leaf switch - 1 interface)
        's10-eth2_datarate', 's10-eth2_throughput', 's10-eth2_packetloss', 's10-eth2_action',
        's10_reward',
    ]

    def __init__(self, filename: str, buffer_size: int = BUFFER_SIZE):
        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = []
        self.file_handle = None
        self.csv_writer = None
        self._initialize_csv()

    def _initialize_csv(self):
        # Create CSV file with headers
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.HEADERS)

    def log_step(self, episode_count: int, step_count: int, traffic_pattern: str,
                 switches_data: Dict[str, Dict], actions_data: Dict[str, List] = None,
                 incidents_data: Dict[str, List] = None):
        # Log a single step
        data_dict = {
            'episode_count': episode_count,
            'step_count': step_count,
            'traffic_pattern': traffic_pattern,
            'total_reward': switches_data.get('total_reward', 0.0)
        }

        # Combine all incidents into single field for quick checking before deployment in Mininet
        incident_interfaces_list = []
        if incidents_data:
            for switch_id, incidents in incidents_data.items():
                if switch_id == 's1':
                    interfaces = ['s1-eth2', 's1-eth3', 's1-eth4']
                    for i, has_incident in enumerate(incidents):
                        if has_incident:
                            incident_interfaces_list.append(interfaces[i])
                else:
                    if incidents and incidents[0]:
                        incident_interfaces_list.append(f'{switch_id}-eth2')

        data_dict['incident_interfaces'] = ', '.join(incident_interfaces_list)

        # Log switch data (rewards and interface metrics)
        for switch_id, switch_data in switches_data.items():
            if isinstance(switch_data, dict):
                data_dict[f'{switch_id}_reward'] = switch_data.get('reward', 0.0)
                interfaces = switch_data.get('interfaces', {})
                for interface_name, interface_data in interfaces.items():
                    data_dict[f'{interface_name}_datarate'] = interface_data.get('datarate', 0.0)
                    data_dict[f'{interface_name}_throughput'] = interface_data.get('throughput', 0.0)
                    data_dict[f'{interface_name}_packetloss'] = interface_data.get('packetloss', 0.0)

        # Log actions
        if actions_data:
            for switch_id, actions in actions_data.items():
                if switch_id == 's1':
                    interfaces = ['s1-eth2', 's1-eth3', 's1-eth4']
                    for i, interface in enumerate(interfaces):
                        if i < len(actions):
                            data_dict[f'{interface}_action'] = actions[i]
                else:
                    interface = f'{switch_id}-eth2'
                    if len(actions) > 0:
                        data_dict[f'{interface}_action'] = actions[0]

        self.buffer.append(data_dict)
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        # Flush buffer to CSV, in marl keeps file handle open
        if not self.buffer:
            return

        # Open file handle on first flush and keep it open
        if self.file_handle is None:
            self.file_handle = open(self.filename, 'a', newline='')
            self.csv_writer = csv.DictWriter(
                self.file_handle,
                fieldnames=self.HEADERS,
                extrasaction='ignore',
                restval=''
            )

        # Write all buffered rows
        for row in self.buffer:
            self.csv_writer.writerow(row)

        # Flush to disk
        self.file_handle.flush()
        self.buffer = []

    def force_flush(self):
        # Force flush buffer to disk
        self._flush_buffer()

    def close(self):
        # Close logger and flush remaining data
        self._flush_buffer()
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.csv_writer = None


# Helper function (the image of the topology in the thesis can be taken as a reference)
def get_parent(switch_id):
    if switch_id == 's1': return None
    if switch_id == 's2': return 's1'
    if switch_id == 's3': return 's1'
    if switch_id == 's4': return 's1'
    if switch_id == 's5': return 's2'
    if switch_id == 's6': return 's3'
    if switch_id == 's7': return 's4'
    if switch_id == 's8': return 's5'
    if switch_id == 's9': return 's6'
    if switch_id == 's10': return 's7'
    return None


class MultiSwitchNetworkEnvironmentSim(gym.Env):
    """Simulated environment for the 10-switch expanded tree topology (digital twin)."""
    def __init__(self, num_switches: int = 10, traffic_type: str = 'decreaseOne', logger: CSVLogger = None,
                 testing: bool = False):
        super().__init__()
        self.switch_ids = ['s%d' % (i + 1) for i in range(num_switches)]
        self.traffic_type = traffic_type
        self.traffic_value = TRAFFIC_PROFILES.get(traffic_type, DEFAULT_DATARATE)
        self.logger = logger
        self.testing = testing  # Flag to distinguish training from testing
        self.max_steps = 50 if testing else 256  # Testing episodes of 50 and Training episodes of 256
        self.interfaces = {}
        self.datarates = {}
        self.throughputs = {}
        self.packet_losses = {}
        self.switch_throughput_efficiencies = {}
        self.episode_step = 0
        self.current_episode = 1
        self.leaf_training_mode = False
        self.switch_rewards = {switch_id: 0.0 for switch_id in self.switch_ids}
        self.incident_interfaces = {}
        self.incident_step_count = {}
        self.action_history = {}

        # Initialization (before and after this comment)
        for switch_id in self.switch_ids:
            self.action_history[switch_id] = {}
            if switch_id == 's1':
                self.interfaces[switch_id] = ['s1-eth2', 's1-eth3', 's1-eth4']
                self.datarates[switch_id] = {iface: DEFAULT_DATARATE for iface in self.interfaces[switch_id]}
                self.throughputs[switch_id] = {iface: 0.0 for iface in self.interfaces[switch_id]}
                self.packet_losses[switch_id] = {iface: 0.0 for iface in self.interfaces[switch_id]}
                self.incident_interfaces[switch_id] = [False] * 3
                for iface in self.interfaces[switch_id]:
                    self.action_history[switch_id][iface] = []
            else:
                iface = f'{switch_id}-eth2'
                self.interfaces[switch_id] = [iface]
                self.datarates[switch_id] = {iface: DEFAULT_DATARATE}
                self.throughputs[switch_id] = {iface: 0.0}
                self.packet_losses[switch_id] = {iface: 0.0}
                self.incident_interfaces[switch_id] = [False]
                self.action_history[switch_id][iface] = []

    # in leaf training we set the throughput of s1-eth2 as the traffic value and train the s2-eth2 interface(although the whole topology is loaded only s2-eth2 interests us
    # helper function to enable or disable leaf training
    def set_leaf_training_mode(self, mode: bool):
        self.leaf_training_mode = mode

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def _apply_routing_scenario(self, iface: str, eth_idx: int) -> float:
        """Return the s1 (root) datarate for one interface under the active
        ROUTING_SCENARIO. Shared by set_traffic_type() and reset() to avoid
        duplicating the scenario logic in both places."""
        if ROUTING_SCENARIO == 0:
            return 50.0 if eth_idx == 2 else self.traffic_value
        elif ROUTING_SCENARIO == 1:
            return 50.0 if eth_idx == 3 else self.traffic_value
        elif ROUTING_SCENARIO == 2:
            return 50.0 if eth_idx == 4 else self.traffic_value
        else:  # ROUTING_SCENARIO == 3
            return 50.0

    def set_traffic_type(self, traffic_type: str):
        if self.traffic_type != traffic_type:
            self.traffic_type = traffic_type
            self.traffic_value = TRAFFIC_PROFILES.get(traffic_type, DEFAULT_DATARATE)
            print(f"*** Traffic type updated to: {traffic_type}")

            # for providing better reward signals and root switch -> different default data rate settings in the beginning
            for iface in self.interfaces['s1']:
                eth_idx = int(iface.split('eth')[1])
                self.datarates['s1'][iface] = self._apply_routing_scenario(iface, eth_idx)

    def reset(self):
        for switch_id in self.switch_ids:
            for iface in self.interfaces[switch_id]:
                if switch_id == 's1':
                    eth_idx = int(iface.split('eth')[1])
                    self.datarates[switch_id][iface] = self._apply_routing_scenario(iface, eth_idx)

                elif getattr(self, "leaf_training_mode", False) and switch_id == 's1':
                    self.datarates[switch_id][iface] = self.traffic_value
                else:
                    self.datarates[switch_id][iface] = DEFAULT_DATARATE
                self.throughputs[switch_id][iface] = 0.0
                self.packet_losses[switch_id][iface] = 0.0
                self.action_history[switch_id][iface].clear()
            if switch_id == 's1':
                self.incident_interfaces[switch_id] = [False] * 3
            else:
                self.incident_interfaces[switch_id] = [False]
        self.incident_step_count.clear()
        # after 256 steps we increase the episode_count
        if self.episode_step >= self.max_steps and not self.testing:
            self.current_episode += 1
            # Force flush buffer when episode ends
            if self.logger:
                self.logger.force_flush()
        elif self.testing and self.episode_step >= 0:
            if self.logger:
                self.logger.force_flush()
        # for new episode reset the step
        self.episode_step = 0
        self.switch_rewards = {switch_id: 0.0 for switch_id in self.switch_ids}
        return self._get_observation()

    def _get_observation(self):
        # root switch observation 10 values (3 datarates, 3 throughputs, 3 packetlosses, incoming traffic = traffic value * 3 branches)
        obs_list = []
        for switch_id in self.switch_ids:
            if switch_id == 's1':
                obs_list.extend([
                    self.datarates[switch_id]['s1-eth2'],
                    self.datarates[switch_id]['s1-eth3'],
                    self.datarates[switch_id]['s1-eth4'],
                    self.throughputs[switch_id]['s1-eth2'],
                    self.throughputs[switch_id]['s1-eth3'],
                    self.throughputs[switch_id]['s1-eth4'],
                    self.packet_losses[switch_id]['s1-eth2'],
                    self.packet_losses[switch_id]['s1-eth3'],
                    self.packet_losses[switch_id]['s1-eth4'],
                    self.traffic_value * 3
                ])
            else:
                # for leaf switches single interface and parent switch with 4 values(datarate, throughput, packetloss, upstream traffic)
                iface = f'{switch_id}-eth2'
                parent = get_parent(switch_id)
                if switch_id == 's2' and getattr(self, "force_s2_upstream_traffic_value", False):
                    upstream_traffic = self.traffic_value
                elif parent is None:
                    upstream_traffic = self.traffic_value
                else:
                    if parent == 's1':
                        idx = ['s2', 's3', 's4'].index(switch_id) if switch_id in ['s2', 's3', 's4'] else 0
                        parent_iface = f's1-eth{2 + idx}'
                    else:
                        parent_iface = f'{parent}-eth2'
                    upstream_traffic = self.throughputs[parent][parent_iface]
                obs_list.extend([
                    self.datarates[switch_id][iface],
                    self.throughputs[switch_id][iface],
                    self.packet_losses[switch_id][iface],
                    upstream_traffic
                ])
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        self.episode_step += 1

        # Force s1 egress datarates to traffic pattern during leaf training
        if getattr(self, "leaf_training_mode", False):
            for interface in self.interfaces["s1"]:
                self.datarates["s1"][interface] = self.traffic_value

        action_index = 0
        actions_data = {}
        incidents_data = {}
        # extract actions for a switch 3 for s1 and 1 for the others
        for switch_id in self.switch_ids:
            if switch_id == 's1':
                n = 3
            else:
                n = 1
            switch_actions = action[action_index:action_index + n]
            action_index += n
            actions_data[switch_id] = list(switch_actions)
            incidents_data[switch_id] = [bool(x) for x in self.incident_interfaces[switch_id]]

            for i, iface in enumerate(self.interfaces[switch_id]):
                # Clear incident after one step (like Mininet)
                if self.incident_interfaces[switch_id][i]:
                    key = (switch_id, i)
                    if key in self.incident_step_count and self.incident_step_count[key] < self.episode_step:
                        self.incident_interfaces[switch_id][i] = False
                        del self.incident_step_count[key]
                # Kept at 0.0 during training; manually raised for incident-response evaluation (see INCIDENT_PROBABILITY note above)
                if random.random() < INCIDENT_PROBABILITY and not self.incident_interfaces[switch_id][i]:
                    self.incident_interfaces[switch_id][i] = True
                    self.incident_step_count[(switch_id, i)] = self.episode_step
                    self.datarates[switch_id][iface] = INCIDENT_DATARATE
                # Apply actions
                if not self.incident_interfaces[switch_id][i] or switch_actions[i] == 2:
                    if switch_actions[i] == 0 and self.datarates[switch_id][iface] > MIN_DATARATE:
                        self.datarates[switch_id][iface] = max(MIN_DATARATE,
                                                               self.datarates[switch_id][iface] - STEP_SIZE)
                    elif switch_actions[i] == 2 and self.datarates[switch_id][iface] < MAX_DATARATE:
                        self.datarates[switch_id][iface] = min(MAX_DATARATE,
                                                               self.datarates[switch_id][iface] + STEP_SIZE)

                self.action_history[switch_id][iface].append(switch_actions[i])
                if len(self.action_history[switch_id][iface]) > 10:
                    self.action_history[switch_id][iface].pop(0)

        self._update_network_performance()
        rewards = self._calculate_rewards()
        self.switch_rewards = rewards
        total_reward = sum(rewards.values()) / len(self.switch_ids)
        done = self.episode_step >= self.max_steps
        switches_data = self._gather_switches_data(rewards)
        switches_data['total_reward'] = total_reward

        info = {
            'switch_rewards': rewards,
            'switches_data': switches_data,
            'actions_data': actions_data,
            'incidents_data': incidents_data,
            'total_reward': total_reward
        }

        # Only log during training (not testing) to prevent double logging
        if self.logger and not self.testing:
            self.logger.log_step(self.current_episode, self.episode_step, self.traffic_type, switches_data,
                                 actions_data, incidents_data)

        return self._get_observation(), total_reward, done, info

    def _update_network_performance(self):
        # Simulate throughput and packet loss
        s1 = 's1'
        for i, iface in enumerate(self.interfaces[s1]):
            branch_traffic = self.traffic_value
            # Optional noise injection for robustness testing; disabled for reported training results
            # tput = min(self.datarates[s1][iface], branch_traffic + np.random.uniform(-2, 2))
            tput = min(self.datarates[s1][iface], branch_traffic)
            self.throughputs[s1][iface] = tput
            self.packet_losses[s1][iface] = max(0.0, 100. * (
                        branch_traffic - tput) / branch_traffic if branch_traffic > 0 else 0)
        self.switch_throughput_efficiencies[s1] = sum(self.throughputs[s1].values()) / (self.traffic_value * 3)
        for switch_id in self.switch_ids[1:]:
            iface = f'{switch_id}-eth2'
            parent = get_parent(switch_id)
            if switch_id == 's2' and getattr(self, "force_s2_upstream_traffic_value", False):
                upstream = self.traffic_value
            elif parent is None:
                upstream = self.traffic_value
            else:
                if parent == 's1':
                    idx = ['s2', 's3', 's4'].index(switch_id) if switch_id in ['s2', 's3', 's4'] else 0
                    parent_iface = f's1-eth{2 + idx}'
                else:
                    parent_iface = f'{parent}-eth2'
                upstream = self.throughputs[parent][parent_iface]
            # Optional noise injection for robustness testing; disabled for reported training results
            # tput = min(self.datarates[switch_id][iface], upstream + np.random.uniform(-2, 2))
            tput = min(self.datarates[switch_id][iface], upstream)
            self.throughputs[switch_id][iface] = tput
            self.packet_losses[switch_id][iface] = max(0.0, 100. * (upstream - tput) / upstream if upstream > 0 else 0)
            self.switch_throughput_efficiencies[switch_id] = tput / upstream if upstream > 0 else 0.1

    def _calculate_rewards(self) -> Dict[str, float]:
        """Compute per-switch reward: balances throughput efficiency against
        packet loss, bandwidth over-allocation, and action flapping."""
        rewards = {}
        for switch_id in self.switch_ids:
            # T_eff_t
            throughput_efficiency = self.switch_throughput_efficiencies[switch_id]
            total_packet_loss = sum(self.packet_losses[switch_id].values())
            # P_loss
            avg_packet_loss = total_packet_loss / len(self.interfaces[switch_id])
            total_bandwidth_limiter = 0
            for iface in self.interfaces[switch_id]:
                if self.packet_losses[switch_id][iface] < 1.0:
                    datarate = self.datarates[switch_id][iface]
                    throughput = self.throughputs[switch_id][iface]
                    # penalizing if more bandwidth is allocated than needed -> the more we over allocate the higher the penalty
                    if datarate > (throughput + 5):
                        limiter = (datarate - throughput) / 100.0
                        total_bandwidth_limiter += limiter
            # W_datarate_t
            avg_bandwidth_limiter = total_bandwidth_limiter / len(self.interfaces[switch_id])
            interfaces_to_check = self.interfaces[switch_id]
            # F_anti_t
            anti_flap_penalty = 0.0
            for iface in interfaces_to_check:
                datarate = self.datarates[switch_id][iface]
                throughput = self.throughputs[switch_id][iface]
                history = self.action_history[switch_id][iface]
                if datarate >= throughput and len(history) > 2:
                    last_two = history[-2:]
                    if (last_two == [2, 0]) or (last_two == [0, 2]):
                        anti_flap_penalty += 0.5
            divisor = len(interfaces_to_check)
            if divisor > 0:
                anti_flap_penalty = anti_flap_penalty / divisor
            # alpha = 10, beta = 6, gamma = 1
            reward = throughput_efficiency - 10 * (
                        avg_packet_loss / 100.0) - 6 * avg_bandwidth_limiter - anti_flap_penalty
            reward = max(-2.0, min(2.0, reward))
            rewards[switch_id] = reward
        return rewards

    def _gather_switches_data(self, rewards: Dict[str, float]):
        # organizing all siwtch interface data for logging
        switches_data = {}
        for switch_id in self.switch_ids:
            switch_data = {
                'reward': rewards[switch_id],
                'interfaces': {}
            }
            for iface in self.interfaces[switch_id]:
                switch_data['interfaces'][iface] = {
                    'datarate': self.datarates[switch_id][iface],
                    'throughput': self.throughputs[switch_id][iface],
                    'packetloss': self.packet_losses[switch_id][iface],
                }
            switches_data[switch_id] = switch_data
        return switches_data


class RootSwitchTrainingWrapperSim(gym.Env):
    """Wrapper for training s1 (root switch with 3 egress interfaces)."""
    def __init__(self, network_env: MultiSwitchNetworkEnvironmentSim):
        super().__init__()
        self.network_env = network_env
        self.switch_id = 's1'
        # action and observation space
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        obs_low = np.array([MIN_DATARATE] * 3 + [0.0] * 3 + [0.0] * 3 + [0.0], dtype=np.float32)
        obs_high = np.array([MAX_DATARATE] * 3 + [MAX_DATARATE] * 3 + [100.0] * 3 + [MAX_DATARATE * 3],
                            dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def seed(self, seed=None):
        return self.network_env.seed(seed)

    def reset(self):
        obs = self.network_env.reset()
        return obs[:ROOT_OBS_DIM]

    def step(self, action):
        full_action = list(action)
        for _ in range(9):  # s2-s10 each get neutral action
            full_action.append(1)
        obs, _, done, info = self.network_env.step(full_action)
        return obs[:ROOT_OBS_DIM], info['switch_rewards']['s1'], done, info


class LeafSwitchTrainingWrapperSim(gym.Env):
    """Wrapper for training leaf switches (1 egress interface each)."""
    def __init__(self, network_env: MultiSwitchNetworkEnvironmentSim, switch_id: str):
        super().__init__()
        self.network_env = network_env
        self.switch_id = switch_id
        # forces s1 to use traffic patterns
        self.network_env.set_leaf_training_mode(True)
        switch_ids = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
        self.switch_index = switch_ids.index(switch_id)
        if self.switch_index == 0:
            self.obs_start = 0
            self.obs_end = 10
        else:
            # Calculates where the switch observation appears in the full observation vector (s1 position 0-10 and s2 10-14,...)
            self.obs_start = 10 + (self.switch_index - 1) * 4
            self.obs_end = self.obs_start + 4
        # every leaf switch has 3 actions each per step to choose from
        self.action_space = spaces.Discrete(3)
        obs_low = np.array([MIN_DATARATE, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([MAX_DATARATE, MAX_DATARATE, 100.0, MAX_DATARATE], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def seed(self, seed=None):
        return self.network_env.seed(seed)

    def reset(self):
        obs = self.network_env.reset()
        return obs[self.obs_start:self.obs_end]

    def step(self, action):
        # for s1 maintain (reminder 0 is decrease, 1 maintain, 2 increase) meaning that s1 holds traffic value
        full_action = [1, 1, 1]
        # action array
        for sid in ['s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']:
            if sid == self.switch_id:
                full_action.append(action)
            else:
                full_action.append(1)
        obs, _, done, info = self.network_env.step(full_action)
        return obs[self.obs_start:self.obs_end], info['switch_rewards'][self.switch_id], done, info


class TrafficPatternCallbackSim(BaseCallback):
    """Custom SB3 callback that rotates traffic patterns after each episode during training."""
    def __init__(self, env, manager, phase, patterns=None, episodes_per_pattern=5, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.manager = manager  # Reference to MultiAgentNetworkManagerSim
        self.phase = phase  # "root" or "leaf"
        self.patterns = patterns or list(TRAFFIC_PROFILES.keys())
        self.episodes_per_pattern = episodes_per_pattern
        self.current_pattern_idx = 0
        self.episodes_in_current_pattern = 0

    def _on_step(self) -> bool:
        # Increment step counter AFTER the step has been processed
        if self.phase == "root":
            self.manager.root_step_counter += 1
            # Check for rotation after incrementing
            self.manager._rotate_log_file_if_needed(self.phase)
        elif self.phase == "leaf":
            self.manager.leaf_step_counter += 1
            # Check for rotation after incrementing
            self.manager._rotate_log_file_if_needed(self.phase)

        return True

    def _on_rollout_end(self) -> None:
        self.episodes_in_current_pattern += 1
        if self.episodes_in_current_pattern >= self.episodes_per_pattern:
            self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.patterns)
            new_pattern = self.patterns[self.current_pattern_idx]
            self.env.set_traffic_type(new_pattern)
            self.episodes_in_current_pattern = 0


class MultiAgentNetworkManagerSim:
    """Orchestrates root/leaf agent training, model persistence, log rotation,
    and multi-agent evaluation for the expanded tree topology."""
    def __init__(self, traffic_type: str = 'decreaseOne'):
        self.traffic_type = traffic_type
        self.root_model_path = "Training_Results/ExpandedTreeTopology/MARL/root_switch_ppo_model.zip"
        self.leaf_model_path = "Training_Results/ExpandedTreeTopology/MARL/leaf_switch_ppo_model.zip"
        self.models_exist = self._check_models_exist()

        # Separate loggers for root and leaf training
        self.root_training_logger = CSVLogger("Training_Results/ExpandedTreeTopology/MARL/MARL_root_training.csv")
        self.leaf_training_logger = CSVLogger("Training_Results/ExpandedTreeTopology/MARL/MARL_leaf_training.csv")
        self.testing_logger = CSVLogger(CSV_FILE_TESTING)

        # File rotation counters
        self.root_file_counter = 1
        self.leaf_file_counter = 1
        self.test_file_counter = 1
        self.root_step_counter = 0
        self.leaf_step_counter = 0
        self.test_step_counter = 0
        self.max_steps_per_file = 512000  # 512k steps per file

        self.env = MultiSwitchNetworkEnvironmentSim(10, traffic_type)
        self.root_switch = 's1'
        self.leaf_switches = ['s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
        self.traffic_patterns = list(TRAFFIC_PROFILES.keys())
        self.episodes_per_pattern = 5

    def _check_models_exist(self):
        return (os.path.isfile(self.root_model_path)) and os.path.isfile(self.leaf_model_path)

    # Rotate log file every 512,000 steps to keep CSV file size manageable (100 full cycles of all 4 traffic patterns, 5,120 steps each)
    def _rotate_log_file_if_needed(self, phase: str):
        # Rotate log files when step limit is reached
        if phase == "root":
            if self.root_step_counter >= self.max_steps_per_file:
                print(f"*** Rotating root training log file at step {self.root_step_counter}")
                self.root_training_logger.close()
                self.root_file_counter += 1
                new_filename = f"Training_Results/ExpandedTreeTopology/MARL/MARL_root_training_part{self.root_file_counter}.csv"
                self.root_training_logger = CSVLogger(new_filename)
                # Update the environment's logger reference
                if hasattr(self, 'env'):
                    self.env.logger = self.root_training_logger
                self.root_step_counter = 0
                print(f"*** Rotated to new root training log file: {new_filename}")

        elif phase == "leaf":
            if self.leaf_step_counter >= self.max_steps_per_file:
                print(f"*** Rotating leaf training log file at step {self.leaf_step_counter}")
                self.leaf_training_logger.close()
                self.leaf_file_counter += 1
                new_filename = f"Training_Results/ExpandedTreeTopology/MARL/MARL_leaf_training_part{self.leaf_file_counter}.csv"
                self.leaf_training_logger = CSVLogger(new_filename)
                # Update the environment's logger reference
                if hasattr(self, 'env'):
                    self.env.logger = self.leaf_training_logger
                self.leaf_step_counter = 0
                print(f"*** Rotated to new leaf training log file: {new_filename}")

        elif phase == "test":
            if self.test_step_counter >= self.max_steps_per_file:
                print(f"*** Rotating testing log file at step {self.test_step_counter}")
                self.testing_logger.close()
                self.test_file_counter += 1
                new_filename = f"Training_Results/ExpandedTreeTopology/MARL/MARL_ExpandedTreeTopology_Simulation_Testing_part{self.test_file_counter}.csv"
                self.testing_logger = CSVLogger(new_filename)
                # Update the environment's logger reference
                if hasattr(self, 'env') and self.env.logger == self.testing_logger:
                    self.env.logger = self.testing_logger
                self.test_step_counter = 0
                print(f"*** Rotated to new testing log file: {new_filename}")

    def train_shared_agent(self, root_timesteps=10240, leaf_timesteps=10240, retrain=False):
        # Only parameters explicitly discussed in the thesis differ from SB3 defaults; remaining parameters set explicitly for experimentation during earlier development
        # verbose is just for info messages during training, is not a learning parameter itself (you can find more information by looking for the SB3 PPO Parameters documentation)
        print("*** Training root switch agent (simulation only)")
        self.env.force_s2_upstream_traffic_value = False
        self.env.logger = self.root_training_logger  # Use root-specific logger

        root_env = DummyVecEnv([lambda: RootSwitchTrainingWrapperSim(self.env)])
        callback = TrafficPatternCallbackSim(self.env, self, "root", self.traffic_patterns, self.episodes_per_pattern)

        if retrain and os.path.isfile(self.root_model_path):
            root_model = PPO.load(self.root_model_path, root_env)
        else:
            root_model = PPO("MlpPolicy", root_env, verbose=1, seed=42,
                             learning_rate=0.0003, n_steps=256, batch_size=64,
                             gamma=0.98, ent_coef=0.01)
        root_model.learn(total_timesteps=root_timesteps, callback=callback)
        root_model.save(self.root_model_path)

        # Close root training logger
        self.root_training_logger.force_flush()

        print("*** Training leaf switch agent (simulation only)")
        self.env.force_s2_upstream_traffic_value = True
        self.env.logger = self.leaf_training_logger  # Use leaf-specific logger

        leaf_env = DummyVecEnv([lambda: LeafSwitchTrainingWrapperSim(self.env, 's2')])
        callback = TrafficPatternCallbackSim(self.env, self, "leaf", self.traffic_patterns, self.episodes_per_pattern)

        if retrain and os.path.isfile(self.leaf_model_path):
            leaf_model = PPO.load(self.leaf_model_path, leaf_env)
        else:
            leaf_model = PPO("MlpPolicy", leaf_env, verbose=1, seed=42,
                             learning_rate=0.0003, n_steps=256, batch_size=64,
                             gamma=0.98, ent_coef=0.01)
        leaf_model.learn(total_timesteps=leaf_timesteps, callback=callback)
        leaf_model.save(self.leaf_model_path)

        # Clean up
        self.leaf_training_logger.force_flush()
        self.env.force_s2_upstream_traffic_value = False
        self.env.logger = None

    def load_models(self):
        root_env = DummyVecEnv([lambda: RootSwitchTrainingWrapperSim(self.env)])
        leaf_env = DummyVecEnv([lambda: LeafSwitchTrainingWrapperSim(self.env, 's2')])
        root_model = PPO.load(self.root_model_path, root_env)
        leaf_model = PPO.load(self.leaf_model_path, leaf_env)
        return root_model, leaf_model

    def deploy_multi_agent_system(self, test_episodes=1):
        # for quick testing of the trained agent in the multiagent network before deployment on mininet
        print("*** Testing trained agents in simulation (multi-agent mode)")
        self.env = MultiSwitchNetworkEnvironmentSim(10, traffic_type=self.traffic_type, logger=self.testing_logger, testing=True)
        root_model, leaf_model = self.load_models()
        self.env.set_leaf_training_mode(False)

        episode_counter = 0
        global_step_counter = 0  # Single global step counter for CSV

        for pattern in self.traffic_patterns:
            self.env.set_traffic_type(pattern)
            for ep in range(test_episodes):
                episode_counter += 1
                self.env.current_episode = episode_counter
                obs = self.env.reset()
                done = False
                episode_step = 0  # Step within this episode
                total_reward = 0.0
                switch_rewards_sum = {sid: 0.0 for sid in self.env.switch_ids}

                while not done:
                    episode_step += 1
                    global_step_counter += 1  # Increment global counter

                    # Get actions from both models
                    root_obs = obs[:ROOT_OBS_DIM]
                    root_action, _ = root_model.predict(root_obs, deterministic=True)
                    leaf_actions = []

                    for i, sid in enumerate(self.leaf_switches):
                        idx = ROOT_OBS_DIM + i * LEAF_OBS_DIM
                        leaf_obs = obs[idx:idx + 4]
                        leaf_action, _ = leaf_model.predict(leaf_obs, deterministic=True)
                        leaf_actions.append(leaf_action)

                    full_action = list(root_action) + leaf_actions
                    obs, reward, done, info = self.env.step(full_action)
                    total_reward += reward

                    for sid in self.env.switch_ids:
                        switch_rewards_sum[sid] += info['switch_rewards'][sid]

                    # Check for log file rotation
                    self.test_step_counter += 1
                    self._rotate_log_file_if_needed("test")

                    # Log once per step with global step counter (testing mode only)
                    self.testing_logger.log_step(
                        episode_counter,
                        global_step_counter,  # Use global counter for CSV
                        self.env.traffic_type,
                        info['switches_data'],
                        info['actions_data'],
                        info['incidents_data']
                    )

                avg_switch_rewards = {sid: switch_rewards_sum[sid] / episode_step for sid in self.env.switch_ids}
                print(
                    f"[Test] Pattern={pattern}, Episode={ep + 1}, Steps={episode_step}, AvgTotalReward={total_reward / episode_step:.4f}")
                for sid in self.env.switch_ids:
                    print(f"  {sid}: avg reward={avg_switch_rewards[sid]:.4f}")

        self.testing_logger.force_flush()

    def set_traffic_type(self, traffic_type: str):
        self.traffic_type = traffic_type
        self.env.set_traffic_type(traffic_type)


# Main experiment function
def run_multi_agent_experiment(root_training_steps, leaf_training_steps, total_testing_episodes, mode, retrain=False):
    manager = MultiAgentNetworkManagerSim()
    if mode == "both" and manager.models_exist and not retrain:
        mode = "test"
    elif mode == "test" and not manager.models_exist:
        mode = "both"

    if mode == "train" or mode == "both":
        manager.train_shared_agent(root_timesteps=root_training_steps, leaf_timesteps=leaf_training_steps, retrain=retrain)
    if mode == "test" or mode == "both":
        manager.deploy_multi_agent_system(test_episodes=total_testing_episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_training_steps', type=int, default=1024000, help="Root switch training steps (default: 1024000)")
    parser.add_argument('--leaf_training_steps', type=int, default=1024000, help="Leaf switch training steps (default: 1024000)")
    parser.add_argument('--total_testing_episodes', type=int, default=1, help="Total test episodes (default: 1)")
    parser.add_argument('--mode', choices=["train", "test", "both"], default="both")
    parser.add_argument('--retrain', action='store_true', help="IF set, retrain models even if they exist")
    args = parser.parse_args()
    start_time = time.time()
    run_multi_agent_experiment(args.root_training_steps, args.leaf_training_steps, args.total_testing_episodes,
                               args.mode, retrain=args.retrain)
    end_time = time.time()
    print(f"*** Total execution time: {end_time - start_time:.2f} seconds")

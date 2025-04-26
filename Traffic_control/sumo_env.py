"""
This SumoIntersectionEnv class takes in parameter a SUMO config file (like an intersection)
and converts it into a gym environment that the reinforce agent can interract with
"""

import traci
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

from scheduler import TlScheduler
scheduler = TlScheduler(tp_min=15, tl_ids=["gneJ1"])


class SumoIntersectionEnv(gym.Env):
    def __init__(self, sumo_cfg_path, use_gui=False, max_steps=500):
        super().__init__()
        self.sumo_cfg = sumo_cfg_path
        self.use_gui = use_gui # for visualization
        self.max_steps = max_steps # max steps in 1 episode
        self.step_count = 0
        self.vmax = 13.89 # max speed 50km/h default
        self.last_tsd = 1e-6
        self.tsd_max = 1e-6  # Initialize to a small non-zero value

        self.sumo_binary = "sumo-gui" if use_gui else "sumo" # choice of visualization
        self.tls_id = "gneJ1"  # traffic light ID ("C" for center traffic light)
        self._start_sumo()


        # Action space = number of phases (SUMO will tell us)
        self.num_phases = len(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases)
        self.action_space = Discrete(self.num_phases)

        # Observation: e.g. waiting time per controlled lane
        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.observation_space = Box(low=0, high=1000, shape=(len(self.controlled_lanes)*3+2,), dtype=np.float32)

        self.prev_tsd = 1e-6  # pour éviter division par zéro

        print("\n🔎 Phase descriptions for traffic light:", self.tls_id)
        for i, phase in enumerate(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases):
            print(f"Phase {i}: duration={phase.duration}, state='{phase.state}'")

    def _start_sumo(self):
        traci.start([
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",  # prevents SUMO from waiting for user input
            "--step-length", "1.0",  # simulation step duration (default: 1s)
            "--quit-on-end"  # cleanly shuts SUMO after last vehicle
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        traci.close()
        self._start_sumo()
        self.step_count = 0
        scheduler.reset()
        return self._get_obs(), {}

    def compute_reward(self):
        tsd = 0.0
        for vid in traci.vehicle.getIDList():
            try:
                v = traci.vehicle.getSpeed(vid)
                delay = max(0.0, 1.0 - (v / self.vmax))
                tsd += delay ** 2
            except:
                pass

        # Update tsd_max
        self.tsd_max = max(self.tsd_max, tsd)

        # Compute normalized reward
        reward = 1.0 - (tsd / self.tsd_max)
        return reward

    def step(self, action):
        # Check if cooldown phase is terminated
        if scheduler.can_act(self.tls_id):
            traci.trafficlight.setPhase(self.tls_id, action)
            scheduler.set_cooldown(self.tls_id)

        # Advance the simulation
        for _ in range(5):  # Simulate 5 steps per decision
            scheduler.step()  # Update cooldowns every real step
            traci.simulationStep()
            self.step_count += 1

        obs = self._get_obs()

        # ---------------------
        # Computing total squared delay (TSD) reward at every step
        # ---------------------

        reward = self.compute_reward()

        terminated = self.step_count >= self.max_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        obs = []
        for lane in self.controlled_lanes:
            waiting_time = traci.lane.getWaitingTime(lane)
            num_vehicles = traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            obs.extend([waiting_time, num_vehicles, mean_speed])

        # Phase actuelle + temps écoulé dans la phase
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        time_in_phase = scheduler.time_in_phase[self.tls_id]

        obs.append(current_phase)
        obs.append(time_in_phase)

        return np.array(obs, dtype=np.float32)

    def render(self):
        pass  # GUI does this

    def get_kpis(self):
        lanes = self.controlled_lanes
        total_waiting_time = sum(traci.lane.getWaitingTime(l) for l in lanes)
        total_delay = 0.0
        total_queue_length = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        total_volume = len(traci.vehicle.getIDList())

        for vid in traci.vehicle.getIDList():
            try:
                v = traci.vehicle.getSpeed(vid)
                delay = 1.0 - (v / self.vmax)
                total_delay += delay
            except:
                pass

        emergency_brakes = traci.simulation.getEmergencyStoppingVehiclesNumber()
        teleports = traci.simulation.getStartingTeleportNumber()

        return {
            "waiting_time": total_waiting_time,
            "delay": total_delay,
            "queue_length": total_queue_length,
            "volume": total_volume,
            "emergency_brakes": emergency_brakes,
            "teleports": teleports
        }

    def close(self):
        traci.close()

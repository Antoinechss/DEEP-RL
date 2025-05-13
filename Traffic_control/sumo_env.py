import traci
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from collections import Counter

from scheduler import TlScheduler
scheduler = TlScheduler(tp_min=15, tl_ids=["gneJ1"])

# ----------------------------
# SUMO Intersection Environment
# ----------------------------

class SumoIntersectionEnv(gym.Env):
    def __init__(self, sumo_cfg_path, use_gui=False, max_steps=500):
        super().__init__()
        self.sumo_cfg = sumo_cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0
        self.vmax = 16.67
        self.phase_usage = Counter()

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.tls_id = "gneJ1"
        self._start_sumo()

        self.num_phases = len(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases)
        self.action_space = Discrete(self.num_phases)

        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.observation_space = Box(low=0, high=1000, shape=(len(self.controlled_lanes)*3+2,), dtype=np.float32)

        print("\n🔎 Phase descriptions for traffic light:", self.tls_id)
        for i, phase in enumerate(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases):
            print(f"Phase {i}: duration={phase.duration}, state='{phase.state}'")

    def _start_sumo(self):
        traci.start([
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--step-length", "1.0",
            "--quit-on-end"
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        traci.close()
        self._start_sumo()
        self.step_count = 0
        scheduler.reset()
        self.phase_usage = Counter()
        return self._get_obs(), {}

    def compute_reward(self):
        teleport_count = traci.simulation.getEndingTeleportNumber()

        tsd = 0.0
        for vid in traci.vehicle.getIDList():
            try:
                v = traci.vehicle.getSpeed(vid)
                delay = max(0.0, 1.0 - (v / self.vmax))
                tsd += delay ** 2
            except:
                pass

        if teleport_count > 0:
            self.teleport_flag = True
            return -1000.0
        else:
            self.teleport_flag = False
            return -tsd

    def step(self, action):
        if scheduler.can_act(self.tls_id):
            traci.trafficlight.setPhase(self.tls_id, action)
            scheduler.set_cooldown(self.tls_id)

        for _ in range(5):
            scheduler.step()
            traci.simulationStep()
            self.step_count += 1

        obs = self._get_obs()
        reward = self.compute_reward()

        terminated = self.step_count >= self.max_steps
        truncated = False

        self.phase_usage[action] += 1

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        obs = []
        for lane in self.controlled_lanes:
            waiting_time = traci.lane.getWaitingTime(lane)
            num_vehicles = traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            obs.extend([waiting_time, num_vehicles, mean_speed])

        current_phase = traci.trafficlight.getPhase(self.tls_id)
        time_in_phase = scheduler.time_in_phase[self.tls_id]

        obs.append(current_phase)
        obs.append(time_in_phase)

        return np.array(obs, dtype=np.float32)

    def render(self):
        pass

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

        # Special events 

        emergency_brakes = traci.simulation.getEmergencyStoppingVehiclesNumber()
        teleports = traci.simulation.getStartingTeleportNumber()

        return {
            "waiting_time": total_waiting_time,
            "delay": total_delay,
            "queue_length": total_queue_length,
            "volume": total_volume,
            "emergency_brakes": emergency_brakes,
            "teleports": teleports,
            "phase_usage": dict(self.phase_usage)
        }

    def close(self):
        traci.close()

import traci

class TlScheduler:
    def __init__(self, tp_min, tl_ids, tp_max=60, queue_threshold=8):
        self.tp_min = tp_min
        self.tp_max = tp_max
        self.tl_ids = tl_ids
        self.cooldowns = {tl_id: 0 for tl_id in tl_ids}
        self.time_in_phase = {tl_id: 0 for tl_id in tl_ids}
        self.queue_threshold = queue_threshold

    def step(self):
        for tl_id in self.tl_ids:
            if self.cooldowns[tl_id] > 0:
                self.cooldowns[tl_id] -= 1
            self.time_in_phase[tl_id] += 1

    def can_act(self, tl_id):
        # Condition 1 : cooldown écoulé ou tp_max atteint
        if self.cooldowns[tl_id] <= 0 or self.time_in_phase[tl_id] >= self.tp_max:
            return True

        # Condition 2 : une autre direction est congestionnée
        if self._is_congested(tl_id):
            return True

        return False

    def set_cooldown(self, tl_id):
        self.cooldowns[tl_id] = self.tp_min
        self.time_in_phase[tl_id] = 0

    def reset(self):
        for tl_id in self.tl_ids:
            self.cooldowns[tl_id] = 0
            self.time_in_phase[tl_id] = 0

    def _is_congested(self, tl_id):
        # Vérifie s’il y a une file avec trop de véhicules
        try:
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            queue_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
            max_queue = max(queue_lengths)
            return max_queue > self.queue_threshold
        except:
            return False

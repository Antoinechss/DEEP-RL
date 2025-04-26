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

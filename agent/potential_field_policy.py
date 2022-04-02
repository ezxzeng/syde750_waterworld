import numpy as np


class PotentialFieldPolicy:
    def __init__(
        self,
        n_sensors=20,
        speed_features=True,
        pursuer_max_accel=0.01,
        obs_weighting=1,
        poison_weighting=1,
        barrier_weighting=1,
        food_weighting=1,
        randomize_angle=False,
        spin_angle=0,
    ):
        self._n_sensors = n_sensors
        self._speed_features = speed_features
        self._pursuer_max_accel = pursuer_max_accel
        self.obs_weighting = obs_weighting
        self.poison_weighting = poison_weighting
        self.barrier_weighting = barrier_weighting
        self.food_weighting = food_weighting
        self.randomize_angle = randomize_angle
        self.spin_angle = spin_angle

        self._angles, self._sensor = self.get_sensors()

    def get_sensors(self, first_angle=0):
        # Generate self._n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self._n_sensors + 1)[:-1]
        # Convert angles to x-y coordinates
        angles += first_angle
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        return angles, sensor_vectors

    def get_angle(self, angle):
        # convert angle from range -sqrt(2) to 2sqrt(2) back to 0, 2pi
        return (angle + np.sqrt(2)) / (3 * np.sqrt(2)) * (2 * np.pi)

    def get_movement_vector(self, observation):
        if self._speed_features:
            obs_dist = observation[0 : self._n_sensors]
            barr_dist = observation[self._n_sensors : (2 * self._n_sensors)]
            food_dist = observation[(2 * self._n_sensors) : (3 * self._n_sensors)]
            food_speed = observation[(3 * self._n_sensors) : (4 * self._n_sensors)]
            poison_dist = observation[(4 * self._n_sensors) : (5 * self._n_sensors)]
            poison_speed = observation[(5 * self._n_sensors) : (6 * self._n_sensors)]
            pursuer_dist = observation[(6 * self._n_sensors) : (7 * self._n_sensors)]
            pursuer_speed = observation[(7 * self._n_sensors) : (8 * self._n_sensors)]
            food_collided = observation[8 * self._n_sensors]
            poison_collided = observation[8 * self._n_sensors + 1]
            angle = observation[8 * self._n_sensors + 2]
        else:
            obs_dist = observation[0 : self._n_sensors]
            barr_dist = observation[self._n_sensors : (2 * self._n_sensors)]
            food_dist = observation[(2 * self._n_sensors) : (3 * self._n_sensors)]
            poison_dist = observation[(3 * self._n_sensors) : (4 * self._n_sensors)]
            pursuer_dist = observation[(4 * self._n_sensors) : (5 * self._n_sensors)]
            food_collided = observation[5 * self._n_sensors]
            poison_collided = observation[5 * self._n_sensors + 1]
            angle = observation[5 * self._n_sensors + 2]

        angle_action = 0
        if self.randomize_angle or self.spin_angle:
            self._angles, self._sensor = self.get_sensors(
                first_angle=self.get_angle(angle)
            )
            if self.spin_angle:
                angle_action = self.spin_angle
            else:
                angle_action = (np.random.random() - 0.5) * 0.1

        repulsion_distances = [
            obs_dist * self.obs_weighting,
            poison_dist * self.poison_weighting,
            barr_dist * self.barrier_weighting,
        ]
        attraction_distances = [food_dist * self.food_weighting]
        max_repulsion_val = np.max(np.concatenate(repulsion_distances))
        repulsion = self.get_force_sensors(
            repulsion_distances,
            replace=(1, max_repulsion_val),
            subtract=max_repulsion_val,
        )
        attraction = self.get_force_sensors(attraction_distances, subtract=0)

        forces = repulsion + attraction

        force_vector = np.sum(forces[:, None] * self._sensor, axis=0)
        force_vector = self.reduce_force_vector(force_vector)

        action_arr = np.append(force_vector, angle_action)

        return action_arr.astype(np.float32)

    def get_force_sensors(self, distances, replace=(1, 0), subtract=0, set_val=None):
        force = np.zeros(self._n_sensors)
        for distance in distances:
            distance[distance == replace[0]] = replace[1]
            if set_val is not None:
                distance[distance != replace[1]] = set_val
            else:
                distance = distance - subtract
            force = force + distance
        return force

    def reduce_force_vector(self, force_vector):
        max_force_val = np.max(np.abs(force_vector))
        if max_force_val > self._pursuer_max_accel:
            force_vector = force_vector / max_force_val * self._pursuer_max_accel

        return force_vector

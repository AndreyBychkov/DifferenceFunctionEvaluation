import numpy as np


class Mesh:
    def __init__(self, function, dt, t_min, t_max, dx, x_min, x_max):
        self._function = function

        self._dt = dt
        self._t_min = t_min
        self._t_max = t_max
        self.n_t = int(np.ceil((t_max - t_min) / dt))

        self._dx = dx
        self._x_min = x_min
        self._x_max = x_max
        self.n_x = int(np.ceil((x_max - x_min) / dx))

        self._mesh = np.zeros((self.n_t, self.n_x))
        self._mesh.fill(np.nan)

    def get_mesh(self):
        return self._mesh.copy()

    def set_initial_condition(self, x_0, x_end, t_0):
        self._mesh[0] = t_0
        self._mesh[:, 0] = x_0
        self._mesh[:, -1] = x_end
        return self

    def evaluate(self):
        for t in range(self.n_t):
            for x in range(self.n_x):
                if np.isnan(self._mesh[t][x]):
                    self._mesh[t][x] = self._function(self._mesh, t, x)

        return self

    def __repr__(self):
        return self._mesh.__repr__()

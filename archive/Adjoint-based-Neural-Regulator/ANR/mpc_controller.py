
import casadi as ca
import numpy as np


class MPC_HOCBF:
    def __init__(self, dt, N, a_min, a_max, w_min, w_max, alpha0, alpha1):
        self.dt = dt
        self.N = N
        self.a_min = a_min
        self.a_max = a_max
        self.w_min = w_min
        self.w_max = w_max
        self.k0 = alpha0
        self.k1 = alpha1

    @staticmethod
    def _as_col(x, rows):
        return ca.reshape(x, rows, 1)

    def dynamics(self, x, u):
        x = self._as_col(x, 4)
        u = self._as_col(u, 2)
        th = x[2]
        v  = x[3]
        a  = u[0]
        w  = u[1]
        return ca.vertcat(v * ca.cos(th), v * ca.sin(th), w, a)

    def rk4(self, x, u):
        x = self._as_col(x, 4)
        u = self._as_col(u, 2)
        dt = self.dt
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + dt/2 * k1, u)
        k3 = self.dynamics(x + dt/2 * k2, u)
        k4 = self.dynamics(x + dt   * k3, u)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def hocbf_value(self, xk, uk, xo, yo, r_eff):
        xk = self._as_col(xk, 4)
        uk = self._as_col(uk, 2)
        x, y, th, v = xk[0], xk[1], xk[2], xk[3]
        a, w = uk[0], uk[1]

        px = x - xo
        py = y - yo

        h = px*px + py*py - r_eff*r_eff

        s = px*ca.cos(th) + py*ca.sin(th)
        t = -px*ca.sin(th) + py*ca.cos(th)

        dh  = 2*v*s
        ddh = 2*s*a + 2*v*t*w + 2*v*v

        return ddh + self.k1*dh + self.k0*h

    @staticmethod
    def angle_wrap(e):
        return ca.atan2(ca.sin(e), ca.cos(e))

    def get_control_input(self, current_state, x_goal, obstacles):
        # Decision variables
        U = ca.MX.sym("U", 2, self.N)

        # Weights
        Q = np.diag([10, 15, 10, 10])
        R = np.diag([1, 1])
        h =50
        H = h * Q

        # Initial state / goal
        xk = self._as_col(current_state, 4)
        xg = self._as_col(x_goal, 4)

        obj = 0
        g = []

        for k in range(self.N):
            uk = U[:, k]

            # cost
            e_pos = xk[0:2] - xg[0:2]
            e_th  = self.angle_wrap(xk[2] - xg[2])
            e_v   = xk[3] - xg[3]
            e = ca.vertcat(e_pos, e_th, e_v)

            obj += ca.mtimes((e).T, Q @ (e)) + ca.mtimes(uk.T, R @ uk)

            # HOCBF constraints (>= 0)
            for obs in obstacles:
                xo = float(obs["xo"])
                yo = float(obs["yo"])
                r_eff = float(obs["r"]) + 0.25 + 0.05
                g.append(self.hocbf_value(xk, uk, xo, yo, r_eff))

            # propagate
            xk = self.rk4(xk, uk)

        # terminal cost
        e_thN = self.angle_wrap(xk[2] - xg[2])
        eN = ca.vertcat(xk[0]-xg[0], xk[1]-xg[1], e_thN, xk[3]-xg[3])
        obj += ca.mtimes(eN.T, H @ eN)

        # NLP
        g_expr = ca.vertcat(*g) if g else ca.vertcat()
        nlp = {"f": obj, "x": ca.reshape(U, -1, 1), "g": g_expr}

        # using ipopt
        opts = {"ipopt.print_level": 0, "print_time": 0}
        #opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.hessian_approximation": "limited-memory",}
        solver = ca.nlpsol("S", "ipopt", nlp, opts)

        # Bounds on U
        lbx = [self.a_min, self.w_min] * self.N
        ubx = [self.a_max, self.w_max] * self.N

       # Bounds on constraints: g >= 0
        num_hocbf = int(g_expr.numel())
        lbg = [0.0] * num_hocbf
        ubg = [ca.inf] * num_hocbf


        # Solve
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        u_opt_traj = sol["x"].full()
        u = u_opt_traj[:2]   # first action
        return u




from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class MPC_ACADOS_RTI:
    """
    Unicycle NMPC controller using acados SQP-RTI with HOCBF obstacle constraints.

    Call get_control_input(...) with the current state, goal state, and obstacle list, then apply the first
    control action.
    """

    def __init__(self,dt,N,a_min,a_max,w_min,w_max, alpha0=3.0, alpha1=3.0, q_diag=None, r_diag=None, terminal_weight=50.0,
        rti_iterations=3, max_obstacles=3,code_export_directory=None,):
        
        self.dt = float(dt)
        self.N = int(N)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.k0 = float(alpha0)
        self.k1 = float(alpha1)
        self.nx = 4
        self.nu = 2
        self.max_obstacles = int(max_obstacles)
        self.np = self.nx + 3 * self.max_obstacles

        self.q_diag = np.array(q_diag if q_diag is not None else [10.0, 15.0, 10.0, 10.0], dtype=float)
        self.r_diag = np.array(r_diag if r_diag is not None else [1.0, 1.0], dtype=float)
        self.terminal_weight = float(terminal_weight)
        self.rti_iterations = max(1, int(rti_iterations))

        if code_export_directory is None:
            code_export_directory = Path(__file__).resolve().parent / "c_generated_code" / "acados_unicycle_hocbf_rti"
        self.code_export_directory = Path(code_export_directory)
        self.code_export_directory.mkdir(parents=True, exist_ok=True)

        self.ocp_solver = self._build_solver()

    @staticmethod
    def wrap_to_pi(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _as_col(x, rows):
        return ca.reshape(x, rows, 1)

    def hocbf_value(self, xk, uk, xo, yo, r_eff):
        xk = self._as_col(xk, 4)
        uk = self._as_col(uk, 2)
        x, y, th, v = xk[0], xk[1], xk[2], xk[3]
        a, w = uk[0], uk[1]

        px = x - xo
        py = y - yo

        h = px * px + py * py - r_eff * r_eff

        s = px * ca.cos(th) + py * ca.sin(th)
        t = -px * ca.sin(th) + py * ca.cos(th)

        dh = 2 * v * s
        ddh = 2 * s * a + 2 * v * t * w + 2 * v * v

        return ddh + self.k1 * dh + self.k0 * h

    def export_model(self):
        model = AcadosModel()
        px = ca.SX.sym("x")
        py = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        v = ca.SX.sym("v")
        x = ca.vertcat(px, py, theta, v)

        a = ca.SX.sym("a")
        omega = ca.SX.sym("omega")
        u = ca.vertcat(a, omega)

        p = ca.SX.sym("p", self.np)
        xdot = ca.SX.sym("xdot", self.nx)
        f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega, a)

        model.name = "unicycle_acados_rti"
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl

        hocbf_constraints = []
        for obs_idx in range(self.max_obstacles):
            obs_offset = self.nx + 3 * obs_idx
            xo = p[obs_offset]
            yo = p[obs_offset + 1]
            r_eff = p[obs_offset + 2]
            hocbf_constraints.append(self.hocbf_value(x, u, xo, yo, r_eff))
        if hocbf_constraints:
            con_h_expr = ca.vertcat(*hocbf_constraints)
            model.con_h_expr = con_h_expr
            model.con_h_expr_0 = con_h_expr

        return model

    def _build_solver(self):
        model = self.export_model()
        ocp = AcadosOcp()
        ocp.model = model

        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        q = np.diag(self.q_diag)
        r = np.diag(self.r_diag)
        px_err = model.x[0] - model.p[0]
        py_err = model.x[1] - model.p[1]
        theta_err = ca.atan2(ca.sin(model.x[2] - model.p[2]), ca.cos(model.x[2] - model.p[2]))
        v_err = model.x[3] - model.p[3]
        state_error = ca.vertcat(px_err, py_err, theta_err, v_err)

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(state_error, model.u)
        ocp.cost.yref = np.zeros(self.nx + self.nu)
        ocp.cost.W = np.block(
            [
                [q, np.zeros((self.nx, self.nu))],
                [np.zeros((self.nu, self.nx)), r],
            ]
        )

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr_e = state_error
        ocp.cost.yref_e = np.zeros(self.nx)
        ocp.cost.W_e = self.terminal_weight * q
        ocp.parameter_values = self._build_parameter_values(np.zeros(self.nx), [])

        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.constraints.lbu = np.array([self.a_min, self.w_min])
        ocp.constraints.ubu = np.array([self.a_max, self.w_max])
        ocp.constraints.idxbu = np.array([0, 1])
        if self.max_obstacles > 0:
            ocp.constraints.lh = np.zeros(self.max_obstacles)
            ocp.constraints.uh = np.full(self.max_obstacles, 1.0e12)
            ocp.constraints.lh_0 = np.zeros(self.max_obstacles)
            ocp.constraints.uh_0 = np.full(self.max_obstacles, 1.0e12)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.nlp_solver_type = "SQP_RTI" 
        ocp.solver_options.nlp_solver_max_iter = self.rti_iterations
        ocp.solver_options.levenberg_marquardt = 1e-3
        ocp.solver_options.print_level = 0
        ocp.code_export_directory = str(self.code_export_directory)

        json_file = self.code_export_directory / "acados_ocp_unicycle_rti.json"
        return AcadosOcpSolver(ocp, json_file=str(json_file))

    def _build_parameter_values(self, x_goal, obstacles):
        x_goal = np.asarray(x_goal, dtype=float).reshape(self.nx,)
        obstacles = obstacles or []
        if len(obstacles) > self.max_obstacles:
            raise ValueError(
                f"Received {len(obstacles)} obstacles, but this acados solver was generated "
                f"for max_obstacles={self.max_obstacles}."
            )

        p = np.zeros(self.np, dtype=float)
        p[: self.nx] = x_goal

        for obs_idx in range(self.max_obstacles):
            obs_offset = self.nx + 3 * obs_idx
            if obs_idx < len(obstacles):
                obs = obstacles[obs_idx]
                p[obs_offset] = float(obs["xo"])
                p[obs_offset + 1] = float(obs["yo"])
                p[obs_offset + 2] = float(obs["r"]) + 0.25 + 0.05
            else:
                p[obs_offset] = 1.0e3
                p[obs_offset + 1] = 1.0e3
                p[obs_offset + 2] = 0.0

        return p

    def set_reference(self, x_goal, obstacles=None):
        p = self._build_parameter_values(x_goal, obstacles)
        for k in range(self.N):
            self.ocp_solver.set(k, "p", p)
            self.ocp_solver.set(k, "yref", np.zeros(self.nx + self.nu))

        self.ocp_solver.set(self.N, "p", p)
        self.ocp_solver.set(self.N, "yref", np.zeros(self.nx))

    def warm_start(self, current_state, x_goal):
        current_state = np.asarray(current_state, dtype=float).reshape(self.nx,)
        current_state[2] = self.wrap_to_pi(current_state[2])
        for k in range(self.N + 1):
            self.ocp_solver.set(k, "x", current_state.copy())
        for k in range(self.N):
            self.ocp_solver.set(k, "u", np.zeros(self.nu))

    def shift_warm_start(self):
        for k in range(self.N - 1):
            self.ocp_solver.set(k, "x", self.ocp_solver.get(k + 1, "x"))
            self.ocp_solver.set(k, "u", self.ocp_solver.get(k + 1, "u"))
        self.ocp_solver.set(self.N - 1, "u", self.ocp_solver.get(self.N - 1, "u"))
        self.ocp_solver.set(self.N, "x", self.ocp_solver.get(self.N, "x"))


    def get_control_input(self, current_state, x_goal, obstacles=None):
        obstacles = obstacles or []
        current_state = np.asarray(current_state, dtype=float).reshape(self.nx,)
        x_goal = np.asarray(x_goal, dtype=float).reshape(self.nx,)
        current_state[2] = self.wrap_to_pi(current_state[2])

        self.ocp_solver.set(0, "lbx", current_state)
        self.ocp_solver.set(0, "ubx", current_state)

        status = 0
        for _ in range(self.rti_iterations):
            self.set_reference(x_goal, obstacles)
            status = self.ocp_solver.solve()
            if status != 0:
                break
        if status != 0:
            self.warm_start(current_state, x_goal)
            for _ in range(self.rti_iterations):
                self.set_reference(x_goal, obstacles)
                status = self.ocp_solver.solve()
                if status != 0:
                    break
        if status != 0:
            raise RuntimeError(f"acados SQP-RTI solver failed with status {status}.")

        u = np.asarray(self.ocp_solver.get(0, "u"), dtype=float).reshape(2, 1)
        self.shift_warm_start()
        return u

    def rk4(self, x, u):
        x = np.asarray(x, dtype=float).reshape(4,)
        u = np.asarray(u, dtype=float).reshape(2,)

        def f(x_val, u_val):
            theta = float(x_val[2])
            v = float(x_val[3])
            a = float(u_val[0])
            omega = float(u_val[1])
            return np.array([v * np.cos(theta), v * np.sin(theta), omega, a], dtype=float)

        k1 = f(x, u)
        k2 = f(x + self.dt / 2.0 * k1, u)
        k3 = f(x + self.dt / 2.0 * k2, u)
        k4 = f(x + self.dt * k3, u)
        x_next = x + self.dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next[2] = self.wrap_to_pi(x_next[2])
        return x_next

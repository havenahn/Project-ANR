import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from HOCBF_SAC.hocbf_qp import HOCBFQPLayer
from HOCBF_SAC.model import DeterministicPolicy, GaussianPolicy, QNetwork
from HOCBF_SAC.utils import hard_update, soft_update, to_tensor


class UnicycleCBFAgent:
    def __init__(self, num_inputs, action_space, env, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.action_space = action_space
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.cbf_mode = args.cbf_mode

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.env = env
        self.cbf_layer = None
        if self.cbf_mode != "off":
            self.cbf_layer = HOCBFQPLayer(env, args)

    def select_action(self, state, dynamics_model, evaluate=False, warmup=False, safe_action=True):
        state = to_tensor(state, torch.FloatTensor, self.device)

        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)

        if warmup:
            batch_size = state.shape[0]
            action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
            for i in range(batch_size):
                action[i] = torch.from_numpy(self.action_space.sample()).to(self.device)
        else:
            if evaluate:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)

        if safe_action:
            final_action = self.get_safe_action(state, action, dynamics_model)
            cbf_action = final_action - action
        else:
            final_action = action
            cbf_action = torch.zeros_like(final_action)

        if expand_dim:
            return final_action.detach().cpu().numpy()[0], cbf_action.detach().cpu().numpy()[0]
        return final_action.detach().cpu().numpy(), cbf_action.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates, dynamics_model):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if self.cbf_mode == "baseline":
                next_state_action = self.get_safe_action(next_state_batch, next_state_action, dynamics_model)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - (self.alpha * next_state_log_pi)
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        if self.cbf_mode == "baseline":
            pi = self.get_safe_action(state_batch, pi, dynamics_model)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(),)

    def save_model(self, output):
        print("Saving models in {}".format(output))
        torch.save(self.policy.state_dict(), "{}/actor.pkl".format(output))
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))

    def load_weights(self, output):
        if output is None:
            return
        print("Loading models from {}".format(output))
        self.policy.load_state_dict(torch.load("{}/actor.pkl".format(output), map_location=self.device))
        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output), map_location=self.device))

    def get_safe_action(self, obs_batch, action_batch, dynamics_model):
        state_batch = dynamics_model.get_state(obs_batch)
        return self.cbf_layer.get_safe_action(state_batch, action_batch)

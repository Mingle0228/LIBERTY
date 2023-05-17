import numpy as np
import torch
from torch import optim
from utils.running_filter.running_filter import ZFilter
from models import cnn_net, mlp_net, mlp_inv_net
from encoder import PixelEncoder,IdentityEncoder
from decoder import PixelDecoder
from utils import select_actions, evaluate_actions
from datetime import datetime
import os
import copy
import math



class ppo_agent:
    def __init__(self, envs, args):
        self.envs = envs 
        self.args = args
        if self.args.env_type == 'atari':
            self.net = cnn_net(envs.action_space.n)
        elif self.args.env_type == 'mujoco':
            self.net = mlp_net(envs.observation_space.shape[0], envs.action_space.shape[0], self.args.dist)
            self.intrinsic_net = mlp_inv_net(envs.observation_space.shape[0])
        self.old_net = copy.deepcopy(self.net)
        if self.args.cuda:
            self.net.cuda()
            self.intrinsic_net.cuda()
            self.old_net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        self.intrinsic_optimizer = optim.Adam(self.intrinsic_net.parameters(), self.args.lr_in, eps=self.args.eps)
        if self.args.env_type == 'mujoco':
            num_states = self.envs.observation_space.shape[0]
            self.running_state = ZFilter((num_states, ), clip=5)
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        if self.args.env_type == 'mujoco':
            self.obs[:] = np.expand_dims(self.running_state(self.envs.reset()), 0)
        else:
            self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.state_optims = None
        if not os.path.exists(self.args.log_data_dir):
            os.mkdir(self.args.log_data_dir)
        self.intrinsic_data_path = '{}/reward_delay_{}'.format(self.args.log_data_dir, self.args.reward_delay_freq)
        if not os.path.exists(self.intrinsic_data_path):
            os.mkdir(self.intrinsic_data_path)
        self.intrinsic_data_path = '{}/seed_{}'.format(self.intrinsic_data_path, self.args.seed)
        if not os.path.exists(self.intrinsic_data_path):
            os.mkdir(self.intrinsic_data_path)


    # start to train the network...
    def learn(self):
        log_data = {}
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        episode_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        delay_step = 0
        delay_rewards = 0
        for update in range(num_updates):
            mb_obs, mb_rewards_ex, mb_actions, mb_dones, mb_values_mix, mb_values_ex, mb_obs_ = [], [], [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    obs_tensor = self._get_tensors(self.obs)
                    v_mix, pis = self.net(obs_tensor)
                    actions = select_actions(pis, self.args.dist, self.args.env_type)
                    actions_tensor = torch.tensor(actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
                    _, v_ex = self.intrinsic_net(obs_tensor)
                if self.args.env_type == 'atari':
                    input_actions = actions 
                else:
                    if self.args.dist == 'gauss':
                        input_actions = actions.copy()
                    elif self.args.dist == 'beta':
                        input_actions = -1 + 2 * actions
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values_mix.append(v_mix.detach().cpu().numpy().squeeze())
                mb_values_ex.append(v_ex.detach().cpu().numpy().squeeze())
                obs_, rewards, dones, _ = self.envs.step(input_actions)
                obs_ = np.expand_dims(self.running_state(obs_), 0)
                mb_obs_.append(np.copy(obs_))
                delay_step += 1
                delay_rewards += rewards
                if dones or delay_step == self.args.reward_delay_freq:
                    rewards = delay_rewards
                    delay_step, delay_rewards = 0, 0
                else:
                    rewards = 0
                if self.args.env_type == 'mujoco':
                    dones = np.array([dones])
                    rewards = np.array([rewards])
                self.dones = dones
                mb_rewards_ex.append(rewards)
                self.obs = obs_
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                        if self.args.env_type == 'mujoco':
                            obs_ = self.envs.reset()
                            self.obs = np.expand_dims(self.running_state(obs_), 0)
                episode_rewards += rewards
                masks = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            mb_obs_ = np.asarray(mb_obs_, dtype=np.float32)
            mb_obs_ = mb_obs_.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_rewards_ex = np.asarray(mb_rewards_ex, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values_mix = np.asarray(mb_values_mix, dtype=np.float32)
            mb_values_ex = np.asarray(mb_values_ex, dtype=np.float32)
            mb_rewards_in = self.compute_rewards(mb_obs, mb_obs_)
            if self.args.env_type == 'mujoco':
                mb_values_mix = np.expand_dims(mb_values_mix, 1)
                mb_values_ex = np.expand_dims(mb_values_ex, 1)
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values_mix, _ = self.net(obs_tensor)
                last_values_mix = last_values_mix.detach().cpu().numpy().squeeze()
                _, last_values_ex = self.intrinsic_net(obs_tensor)
                last_values_ex = last_values_ex.detach().cpu().numpy().squeeze()
            mb_values_mix_next = np.zeros_like(mb_values_mix)
            mb_values_mix_next[:-1] = mb_values_mix[1:] * (1.0 - mb_dones[1:])
            mb_values_mix_next[-1] = last_values_mix * (1 - self.dones)
            td_mix = self.args.gamma * mb_values_mix_next - mb_values_mix
            mb_advs_mix = np.zeros_like(mb_rewards_ex)
            mb_advs_ex = np.zeros_like(mb_rewards_ex)
            mb_rewards_mix = self.args.r_ext_coef * mb_rewards_ex + self.args.r_in_coef * mb_rewards_in
            lastgaelam_mix, lastgaelam_ex = 0, 0
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues_mix = last_values_mix
                    nextvalues_ex = last_values_ex
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues_mix = mb_values_mix[t + 1]
                    nextvalues_ex = mb_values_ex[t + 1]
                delta_mix = mb_rewards_mix[t] + self.args.gamma * nextvalues_mix * nextnonterminal - mb_values_mix[t]
                delta_ex = mb_rewards_ex[t] + self.args.gamma * nextvalues_ex * nextnonterminal - mb_values_ex[t]
                mb_advs_mix[t] = lastgaelam_mix = delta_mix + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam_mix
                mb_advs_ex[t] = lastgaelam_ex = delta_ex + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam_ex
            mb_returns_mix = mb_advs_mix + mb_values_mix
            mb_returns_ex = mb_advs_ex + mb_values_ex
            mb_rewards_ex = mb_rewards_ex.swapaxes(0, 1).flatten()
            mb_rewards_in = mb_rewards_in.swapaxes(0, 1).flatten()
            td_mix = td_mix.swapaxes(0, 1).flatten()
            mb_dones = mb_dones.swapaxes(0, 1).flatten()
            mb_values_mix = mb_values_mix.swapaxes(0, 1).flatten()
            self.old_net.load_state_dict(self.net.state_dict())
            pl, vl, ent = self._update_network(mb_obs, mb_actions, mb_returns_mix, mb_returns_ex, mb_advs_mix, mb_advs_ex, \
                    mb_rewards_in, mb_rewards_ex, td_mix, mb_dones, mb_values_mix, mb_obs_)
            if update % self.args.display_interval == 0:
                print('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f},  PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean(), final_rewards.min(), final_rewards.max(),  pl, vl, ent))
                if self.args.env_type == 'atari':
                    torch.save(self.net.state_dict(), self.model_path + '/model.pt')
                else:
                    # for the mujoco, we also need to keep the running mean filter!
                    torch.save([self.net.state_dict(), self.running_state, self.intrinsic_net.state_dict()], self.model_path + '/model.pt')
            # save log data
            log_data[update] = {'frames': (update + 1)*self.args.nsteps*self.args.num_workers, 'rewards_mean': final_rewards.mean(), \
                    'rewards_in': np.mean(mb_rewards_in), 'rewards_ex': np.mean(mb_rewards_ex)}
            torch.save(log_data, '{}/{}.pt'.format(self.intrinsic_data_path, self.args.env_name))

    def compute_rewards(self, obs, obs_, requires_grad=False):
        """
        this process will be super time-consuming
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        obs_next_tensor = torch.tensor(obs_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        if not requires_grad:
            with torch.no_grad():
                # get the feature maps of the observation
                feats_in, _ = self.intrinsic_net(obs_tensor)
                feats_in_next, _ = self.intrinsic_net(obs_next_tensor)
        else:
            feats_in, _ = self.intrinsic_net(obs_tensor)
            feats_in_next, _ = self.intrinsic_net(obs_next_tensor)
        # calculate the cosine distance
        if self.args.metric_type == "no_inv":
            """
            using a more efficient to calculate the diversity between two adjancent states
            """
            cos_dist = torch.nn.functional.cosine_similarity(feats_in, feats_in_next, dim=1)
            # calculate the intrinsic reward - det(L)
            rewards_in = 1 - cos_dist.pow(2)
        else:
            """
            the normal way to calculate the diversity, will be slower
            they should get the same intrinsic rewards.
            """
            # normalize the  vector
            feats_in = torch.nn.functional.normalize(feats_in, p=2, dim=1)
            feats_in_next = torch.nn.functional.normalize(feats_in_next, p=2, dim=1)
            # from the feature vectors
            feats_in = feats_in.unsqueeze(1)
            feats_in_next = feats_in_next.unsqueeze(1)
            feat_vec = torch.cat([feats_in, feats_in_next], dim=1)
            feat_vec_T = torch.transpose(feat_vec, 1, 2)
            # cal
            kernel_mat = torch.matmul(feat_vec, feat_vec_T)        
            rewards_in = torch.det(kernel_mat)
        return rewards_in.unsqueeze(-1) if requires_grad else rewards_in.unsqueeze(-1).detach().cpu().numpy()


    def _update_network(self, obs, actions, returns_mix, returns_ex, advantages_mix, advantages_ex, rewards_in, rewards_ex, td_mix, dones, values_mix, obs_):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                """
                here I will calculate the coefficient mat - no other way to optimize this process, 

                use matrix computation is already the best way to calculate the advantages of the 
                """
                coef_mat = np.zeros((nbatch_train, obs.shape[0]), dtype=np.float32)
                for i in range(nbatch_train):
                    coef = 1.0
                    for j in range(mbinds[i], obs.shape[0]):
                        if j > mbinds[i] and (dones[j] or j % self.args.nsteps == 0):
                            break
                        coef_mat[i][j] = coef
                        coef *= self.args.gamma * self.args.tau
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                actions_tensor = torch.tensor(actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                r_in_tensor = self.compute_rewards(obs, obs_, requires_grad=True)
                r_ex_tensor = torch.tensor(rewards_ex, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
                td_mix_tensor = torch.tensor(td_mix, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
                coef_mat_tensor = torch.tensor(coef_mat, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                delta_mix = self.args.r_in_coef * r_in_tensor + self.args.r_ext_coef * r_ex_tensor + td_mix_tensor
                adv_mix = torch.matmul(coef_mat_tensor, delta_mix)
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_values_mix = values_mix[mbinds]
                mb_advs_ex = advantages_ex[mbinds]
                mb_returns_ex = returns_ex[mbinds]
                mb_obs = self._get_tensors(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                mb_values_mix = torch.tensor(mb_values_mix, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                mb_advs_ex = torch.tensor(mb_advs_ex, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                mb_returns_ex = torch.tensor(mb_returns_ex, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                returns_mix = adv_mix + mb_values_mix
                adv_mix = (adv_mix - adv_mix.mean().detach()) / (adv_mix.std().detach() + 1e-8)
                mb_values, pis = self.net(mb_obs)
                value_loss = (returns_mix - mb_values).pow(2).mean()
                with torch.no_grad():
                    _, old_pis = self.old_net(mb_obs)
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions, self.args.dist, self.args.env_type)
                    old_log_prob = old_log_prob.detach()
                log_prob, ent_loss = evaluate_actions(pis, mb_actions, self.args.dist, self.args.env_type)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                surr1 = prob_ratio * adv_mix
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * adv_mix
                policy_loss = -torch.min(surr1, surr2).mean()
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                self.optimizer.zero_grad()
                self.intrinsic_optimizer.zero_grad()
                grads = torch.autograd.grad(total_loss, self.net.parameters(), create_graph=True)
                net_new = copy.deepcopy(self.net)
                for (_, param), grad in zip(self.net.named_parameters(), grads):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                        param.grad.data.copy_(grad.data)
                    else:
                        param.grad.data.copy_(grad.data)
                self.optimizer.step()
                if self.state_optims is None:
                    self.state_optims = self.optimizer.state_dict()['state'].values()
                    self.init_optim = True
                beta1, beta2 = 0.9, 0.999
                for (_, param), grad, state_optim in zip(net_new.named_parameters(), grads, self.state_optims):
                    """
                    it's the initial state, therefore, we set the exp_avg and exp_std to 0
                    """
                    if self.init_optim:
                        exp_avg = torch.zeros_like(param)
                        exp_avg_sq = torch.zeros_like(param)
                    else:
                        exp_avg = state_optim['exp_avg'].clone()
                        exp_avg_sq = state_optim['exp_avg_sq'].clone()
                    bias_corr1 = 1 - beta1 ** state_optim['step']
                    bias_corr2 = 1 - beta2 ** state_optim['step']
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad.detach(), grad.detach())
                    step_size = self.args.lr / bias_corr1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(self.args.eps)
                    param.requires_grad = False
                    param.addcdiv_(-step_size, exp_avg, denom)
                self.init_optim = False
                self.state_optims = self.optimizer.state_dict()['state'].values()
                mb_advs_ex = (mb_advs_ex - mb_advs_ex.mean()) / (mb_advs_ex.std() + 1e-8)
                _, pis_new = net_new(mb_obs)
                new_log_prob, _ = evaluate_actions(pis_new, mb_actions, self.args.dist, self.args.env_type)
                ratio_new = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio_new * mb_advs_ex
                surr2 = torch.clamp(ratio_new, 1 - self.args.clip, 1 + self.args.clip) * mb_advs_ex
                in_policy_loss = -torch.min(surr1, surr2).mean()
                _, mb_values_ex = self.intrinsic_net(mb_obs)
                in_value_loss = (mb_returns_ex - mb_values_ex).pow(2).mean()
                in_total_loss = in_policy_loss + self.args.vloss_coef * in_value_loss
                eta_grads = torch.autograd.grad(in_total_loss, self.intrinsic_net.parameters())
                for param, grad in zip(self.intrinsic_net.parameters(), eta_grads):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                        param.grad.data.copy_(grad.data)
                    else:
                        param.grad.data.copy_(grad.data)
                self.intrinsic_optimizer.step()

        return policy_loss.item(), value_loss.item(), ent_loss.item()

    def _get_tensors(self, obs):
        if self.args.env_type == 'atari':
            obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor

    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

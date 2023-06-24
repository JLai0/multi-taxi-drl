import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
import os
import random
import string

import argparse

from tqdm import tqdm
from commons import create_env_map

from multi_taxi_env import MultiTaxiEnv


written_actions = ['South', 'North', 'East', 'West', 'Wait', 'Pickup', 'Dropoff']


def init_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rs', '--random_seed', type=int, default=42)
    parser.add_argument('-bl', '--use_baseline', type=bool, help='Use the baseline environment.', default=False)
    parser.add_argument('-nc', '--number_of_columns', type=int, help='The number of columns in the environment.', default=5)
    parser.add_argument('-nr', '--number_of_rows', type=int, help='The number of rows in the environment.', default=5)
    parser.add_argument('-ab', '--amount_of_borders', type=float, help='The amount of borders in the environment.', default=0.0)
    parser.add_argument('-nt', '--number_of_taxis', type=int, help='The number of taxis in the environment.', default=1)
    parser.add_argument('-np', '--number_of_passengers', type=int, help='The number of passengers in the environment.', default=1)
    parser.add_argument('-n', '--name', type=str, help='The name of the run.', default='None')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--train_n', type=int, default=10)
    parser.add_argument('--test_n', type=int, default=100)
    parser.add_argument('-g', '--gamma', type=float, default=.9)
    parser.add_argument('--n_step', type=int, help='#steps to look ahead', default=3)
    parser.add_argument('--target_freq', type=int, help='Target update frequency', default=320)
    parser.add_argument('--buffer_size', type=int, help='Size of replay buffer', default=20000)
    parser.add_argument('--eps_train', type=float, help='Epsilon for epsilon-greedy exploration.', default=.1)
    parser.add_argument('--eps_test', type=float, help='Epsilon for epsilon-greedy exploration.', default=.05)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--steps_per_collect', type=int, default=10)
    parser.add_argument('-nn', '--number_of_nodes', nargs='+', type=int, default=[512, 512, 512, 512])
    return parser.parse_args()

class BLDuelingDQN(nn.Module):
  """Dueling DQN that computes Q-values through value and advantage."""
  def __init__(self, state_shape, action_shape, number_of_nodes):
    super(BLDuelingDQN, self).__init__()
    self.emb = nn.Embedding(state_shape, 4)
    self.fc1 = nn.Linear(4, number_of_nodes[0])
    self.fc_h_v = nn.Linear(number_of_nodes[0], number_of_nodes[1])
    self.fc_h_a = nn.Linear(number_of_nodes[0], number_of_nodes[1])
    self.fc_z_v = nn.Linear(number_of_nodes[1], 1)
    self.fc_z_a = nn.Linear(number_of_nodes[1], action_shape)
    self.explain = False
    self.visits = torch.zeros(500)

  def forward(self, obs, state=None, info={}):
    if not isinstance(obs, torch.Tensor):
      obs = torch.tensor(obs, dtype=torch.int64)
    self.visits[obs] += 0.00001
    x = self.emb(obs)
    x = F.leaky_relu(self.fc1(x.squeeze(dim=1)))
    value = self.fc_z_v(F.leaky_relu(self.fc_h_v(x)))  # Value stream
    advantage = self.fc_z_a(F.leaky_relu(self.fc_h_a(x)))  # Advantage stream

    Q_values = value + advantage - advantage.mean(1, keepdim=True)  # Combine two streams of DQN
    if self.explain:
      return Q_values, state, value, advantage
    else:
      return Q_values, state
    # TODO: Don't use embedding
    
class MultiTaxiDuelingDQN(nn.Module):

  """Dueling DQN that computes Q-values through value and advantage."""
  def __init__(self, state_shape, action_shape, number_of_nodes):
    super(MultiTaxiDuelingDQN, self).__init__()
    self.fc1 = nn.Linear(state_shape, number_of_nodes[0])
    self.fc2 = nn.Linear(number_of_nodes[0], number_of_nodes[1])
    self.fc_h_v = nn.Linear(number_of_nodes[1], number_of_nodes[2])
    self.fc_h_a = nn.Linear(number_of_nodes[1], number_of_nodes[2])
    self.fc_z_v = nn.Linear(number_of_nodes[2], 1)
    self.fc_z_a = nn.Linear(number_of_nodes[2], action_shape)
    self.explain = False
    self.state_shape = state_shape

  def forward(self, obs, state=None, info={}):
    x = torch.tensor(list(map(self.flatten_state, obs)), dtype=torch.float32)
    x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    value = self.fc_z_v(F.leaky_relu(self.fc_h_v(x)))  # Value stream
    advantage = self.fc_z_a(F.leaky_relu(self.fc_h_a(x)))  # Advantage stream
    q_values = value + advantage - advantage.mean(1, keepdim=True)  # Combine two streams of DQN
    return q_values, state

  def flatten_state(self, state):
      result = []
      [(result.append(t.loc.x), result.append(t.loc.y), result.append(t.pass_idx)) for t in state.taxis]
      [(result.append(p.loc.x), result.append(p.loc.y), result.append(p.dest.x), result.append(p.dest.y)) for p in state.passengers]
      return result

def generate_name(args):

  idx = ''.join([random.choice(string.ascii_letters) for i in range(4)]).upper()
  if args.use_baseline:
    return f"{idx}_BL"
  else:
    return f"{idx}_nc{args.number_of_columns}_nr{args.number_of_rows}_nt{args.number_of_taxis}_np{args.number_of_passengers}"

def evaluate(env, policy):
  acc_rewards, js, deliveries, action_stds, nof_runs = [], [], [], [], 25
  for i in (pbar := tqdm(range(nof_runs))):
    pbar.set_description(f'Step: {i}')
    j, terminated, truncated, acc_reward, actions = 0, False, False, 0, []
    obs, _ = env.reset()
    while (not terminated) and (not truncated):
      j += 1
      q_values, _ = policy.model([obs])
      action = q_values.argmax(1).item()
      obs, reward, terminated, truncated, info = env.step(action)
      acc_reward += reward
      actions.append(action)
    
    acc_rewards.append(acc_reward), js.append(j), deliveries.append(terminated)
    action_stds.append(np.asarray(actions).std())

  return {
    'test/acc_rewards': np.asarray(acc_rewards).mean(),
    'test/steps': np.asarray(js).mean(),
    'test/deliveries': np.asarray(deliveries).sum() / len(deliveries),
    'test/actions_stds': np.asarray(action_stds).mean(),
  }

def log_multi_taxi_dueling_dqn_weights_and_biases(writer, policy, epoch):
  writer.add_histogram('model/fc1/weight', policy.model.fc1.weight, global_step=epoch)
  writer.add_histogram('model/fc1/bias', policy.model.fc1.bias, global_step=epoch)
  writer.add_histogram('model/fc2/weight', policy.model.fc2.weight, global_step=epoch)
  writer.add_histogram('model/fc2/bias', policy.model.fc2.bias, global_step=epoch)
  writer.add_histogram('model/fc_h_v/weight', policy.model.fc_h_v.weight, global_step=epoch)
  writer.add_histogram('model/fc_h_v/bias', policy.model.fc_h_v.bias, global_step=epoch)
  writer.add_histogram('model/fc_h_a/weight', policy.model.fc_h_a.weight, global_step=epoch)
  writer.add_histogram('model/fc_h_a/bias', policy.model.fc_h_a.bias, global_step=epoch)
  writer.add_histogram('model/fc_z_v/weight', policy.model.fc_z_v.weight, global_step=epoch)
  writer.add_histogram('model/fc_z_v/bias', policy.model.fc_z_v.bias, global_step=epoch)
  writer.add_histogram('model/fc_z_a/weight', policy.model.fc_z_a.weight, global_step=epoch)
  writer.add_histogram('model/fc_z_a/bias', policy.model.fc_z_a.bias, global_step=epoch)


def main():

    args = init_args_parser()
    args.name = generate_name(args)
    print(args.name)

    random.seed(args.random_seed), torch.manual_seed(args.random_seed), np.random.seed(args.random_seed)
    
    writer = SummaryWriter(f'./logs/{args.name}')
    logger = ts.utils.TensorboardLogger(writer)

    if args.use_baseline:
      task = 'Taxi-v3'
      train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(args.train_n)])
      test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(args.test_n)])
      env = gym.make(task, render_mode='ansi')
      state_shape = env.observation_space.shape or env.observation_space.n
      action_shape = env.action_space.shape or env.action_space.n
      net = BLDuelingDQN(state_shape=state_shape, action_shape=action_shape, number_of_nodes=args.number_of_nodes)
    else:
      env_map = create_env_map(args.number_of_columns, args.number_of_rows, args.amount_of_borders)
      train_envs = ts.env.DummyVectorEnv([lambda: MultiTaxiEnv(args, env_map) for _ in range(args.train_n)])
      test_envs = ts.env.DummyVectorEnv([lambda: MultiTaxiEnv(args, env_map) for _ in range(args.test_n)])
      env = MultiTaxiEnv(args, env_map)
      net = MultiTaxiDuelingDQN(state_shape=env.observation_space, action_shape=7**args.number_of_taxis, number_of_nodes=args.number_of_nodes)

    # net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # policy = ts.policy.DQNPolicy(
    #   model=net, 
    #   optim=optim, 
    #   discount_factor=args.gamma, # TODO: Try .99
    #   estimation_step=args.n_step, 
    #   target_update_freq=args.target_freq
    # )
    policy = ts.policy.DQNPolicy(
      model=net, 
      optim=optim, 
      discount_factor=.99,
      estimation_step=1, 
      target_update_freq=0,
    )
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(args.buffer_size, args.train_n), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

    def save_checkpoint_fn(epoch, env_step, gradient_step):
      if epoch % 25 == 0:
        if args.use_baseline:
          pass
        else:
          log_multi_taxi_dueling_dqn_weights_and_biases(writer, policy, epoch)
  
      if epoch % 50 == 0:
        print(args.name)
        metric_dict = evaluate(env, policy)
        for key, value in metric_dict.items():
          writer.add_scalar(key, value, epoch)

      if epoch % 500 == 0:
        save_path = os.path.join('./checkpoints/', f'{args.name}_checkpoint_{epoch}.pth')
        torch.save(
          {
            'config': vars(args),
            'epoch': epoch,
            'env_step': env_step,
            'gradient_step': gradient_step,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
          }, save_path)

        return save_path
      
    def save_best_fn(policy):
      torch.save(policy.state_dict(), os.path.join(f'./best_model/{args.name}'))

    result = ts.trainer.offpolicy_trainer(
      policy=policy, 
      train_collector=train_collector, 
      test_collector=test_collector, 
      max_epoch=args.epochs, 
      step_per_epoch=args.steps_per_epoch, 
      step_per_collect=args.steps_per_collect,
      episode_per_test=args.test_n, 
      batch_size=args.batch_size, 
      update_per_step=1/args.steps_per_collect,
      train_fn=lambda epoch, env_step: policy.set_eps(args.eps_train),
      test_fn=lambda epoch, env_step: policy.set_eps(args.eps_test),
      stop_fn=lambda mean_rewards: mean_rewards >= 30,
      save_best_fn=save_best_fn,
      save_checkpoint_fn=save_checkpoint_fn,
      # resume_from_log
      # reward_metric
      verbose=True,
      show_progress=True,
      # test_in_train
      logger=logger,
    )
    print(f'Finished training! Use {result["duration"]}')
    torch.save(policy.state_dict(), f'./models/{args.name}')

if __name__ == '__main__':
    main()

# - How to learn a neural network based policy?
# - Joint policy?
# - Each agent one or all agents the same policy?
# - Cooperative or adversarial or mixed? 
# - Maybe simply do the iteration in the network? 
# - Maybe it works directly with tianshous approach - only the state returned by the environment has to be changed
# - Decentralised vs centralized (parameter sharing)?
# - Or share the state? 

# torch.nn-Embedding
# - A simple lookup table that stores embeddings of a ficed dictionary and size. 
# - This module is often used to store word embeddings and retrieve them using indices. The input to the module 
#     is a list of indices, and the output is the corresponding word embeddings.
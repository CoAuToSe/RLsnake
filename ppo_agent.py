# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import PPOModel
from game import SnakeGameAI, Direction, Point, TAILLE_BLOC
import numpy as np

class PPOAgent:
    def __init__(self, input_size, hidden_size, output_size, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.model = PPOModel(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.policy_old = PPOModel(input_size, hidden_size, output_size)
        self.policy_old.load_state_dict(self.model.state_dict())

        self.MseLoss = nn.MSELoss()

        # Stockage des trajectoires
        self.memory = []
        
        self.episode = 0

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            policy_logits, _ = self.policy_old(state)
            policy = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item()

    def store_transition(self, state, action, reward, next_state, done, action_logprob, value):
        self.memory.append((state, action, reward, next_state, done, action_logprob, value))

    def train(self):
        # Convertir la mémoire en tenseurs
        states, actions, rewards, next_states, dones, old_logprobs, values = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.tensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        old_logprobs = torch.FloatTensor(old_logprobs)
        values = torch.FloatTensor(values)

        # Calcul des retours et des avantages
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        advantages = returns - values

        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Optimisation
        for _ in range(self.K_epochs):
            # Recalculer les log-probabilités et les valeurs
            policy_logits, new_values = self.model(states)
            policy = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(policy)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Calcul du ratio
            ratios = torch.exp(new_logprobs - old_logprobs)

            # Calcul des pertes
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.MseLoss(new_values.squeeze(), returns)
            loss_entropy = -0.01 * entropy.mean()

            loss = loss_actor + 0.5 * loss_critic + loss_entropy

            # Mise à jour des paramètres
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Mettre à jour la politique ancienne
        self.policy_old.load_state_dict(self.model.state_dict())
        # Vider la mémoire
        self.memory = []
        
        
    def get_state(self, game):
        tête = game.serpent[0]
        point_l = Point(tête.x - TAILLE_BLOC, tête.y)
        point_r = Point(tête.x + TAILLE_BLOC, tête.y)
        point_u = Point(tête.x, tête.y - TAILLE_BLOC)
        point_d = Point(tête.x, tête.y + TAILLE_BLOC)

        dir_l = game.direction == Direction.GAUCHE
        dir_r = game.direction == Direction.DROITE
        dir_u = game.direction == Direction.HAUT
        dir_d = game.direction == Direction.BAS

        état = [
            # Danger devant
            (dir_r and game._collision(point_r)) or
            (dir_l and game._collision(point_l)) or
            (dir_u and game._collision(point_u)) or
            (dir_d and game._collision(point_d)),

            # Danger à droite
            (dir_r and game._collision(point_d)) or
            (dir_l and game._collision(point_u)) or
            (dir_u and game._collision(point_r)) or
            (dir_d and game._collision(point_l)),

            # Danger à gauche
            (dir_r and game._collision(point_u)) or
            (dir_l and game._collision(point_d)) or
            (dir_u and game._collision(point_l)) or
            (dir_d and game._collision(point_r)),

            # Direction de déplacement
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Nourriture
            game.nourriture.x < game.tête.x,  # Nourriture à gauche
            game.nourriture.x > game.tête.x,  # Nourriture à droite
            game.nourriture.y < game.tête.y,  # Nourriture en haut
            game.nourriture.y > game.tête.y   # Nourriture en bas
        ]
        return np.array(état, dtype=int)
        
    def save(self, filename='ppo_agent.pth'):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.episode,  # Si vous suivez le numéro d'épisode dans l'agent
            'memory': self.memory
        }
        torch.save(checkpoint, filename)
        # print(f"Agent sauvegardé dans {filename}")
        
    def load(self, filename='ppo_agent.pth'):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode = checkpoint.get('episode', 0)
        self.memory = checkpoint.get('memory', [])
        print(f"Agent chargé depuis {filename}, à partir de l'épisode {self.episode}")

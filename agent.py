# agent.py
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinQNet

TAILLE_BLOC = 20
MAX_MEMOIRE = 100_000
TAILLE_BATCH = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memoire = deque(maxlen=MAX_MEMOIRE)  # Popleft()
        self.model = LinQNet(11, 256, 3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()

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

            # Position de la nourriture par rapport à la tête
            game.nourriture.x < game.tête.x,  # Nourriture à gauche
            game.nourriture.x > game.tête.x,  # Nourriture à droite
            game.nourriture.y < game.tête.y,  # Nourriture en haut
            game.nourriture.y > game.tête.y   # Nourriture en bas
        ]

        return np.array(état, dtype=int)

    def remember(self, état, action, reward, next_state, done):
        self.memoire.append((état, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memoire) > TAILLE_BATCH:
            mini_sample = random.sample(self.memoire, TAILLE_BATCH)
        else:
            mini_sample = self.memoire

        états, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(états, actions, rewards, next_states, dones)

    def train_step(self, états, actions, rewards, next_states, dones):
        états = torch.tensor(np.array(états), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)

        # Pred Q values avec l'état courant
        pred = self.model(états)

        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][torch.argmax(actions[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, état):
        # return self.get_action_rand(état)
        # return self.get_action_ech_rand(état)
        return self.get_action_dec_exp(état)
        # return self.get_action_range(état)

    def get_action_rand(self, état):
        # Mouvement aléatoire : tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            état0 = torch.tensor(état, dtype=torch.float)
            prediction = self.model(état0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def get_action_ech_rand(self, état):
        # Mouvement aléatoire : tradeoff exploration / exploitation
        self.epsilon = max(0.01, 0.1 - (self.n_games * 0.001))
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            état0 = torch.tensor(état, dtype=torch.float)
            prediction = self.model(état0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_action_dec_exp(self, état):
        # Mouvement aléatoire : tradeoff exploration / exploitation
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.005
        self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * self.n_games)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            état0 = torch.tensor(état, dtype=torch.float)
            prediction = self.model(état0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


    def get_action_range(self, état):
        # Mouvement aléatoire : tradeoff exploration / exploitation
        self.epsilon = max(0.01, min(1, 1 - np.log10((self.n_games + 1) / 25)))
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            état0 = torch.tensor(état, dtype=torch.float)
            prediction = self.model(état0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save(self, file_name='agent.pth'):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            # Vous pouvez ajouter d'autres variables si nécessaire
        }
        torch.save(checkpoint, file_name)

    def load(self, file_name='agent.pth'):
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint.get('n_games', 0)
        self.epsilon = checkpoint.get('epsilon', 0)
        # Charger d'autres variables si vous les avez sauvegardées
    
#     def save(self, file_name='agent.pth'):
#     checkpoint = {
#         'model_state_dict': self.model.state_dict(),
#         'optimizer_state_dict': self.optimizer.state_dict(),
#         'n_games': self.n_games,
#         'epsilon': self.epsilon,
#         'memoire': self.memoire
#     }
#     torch.save(checkpoint, file_name)

# def load(self, file_name='agent.pth'):
#     checkpoint = torch.load(file_name)
#     self.model.load_state_dict(checkpoint['model_state_dict'])
#     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     self.n_games = checkpoint.get('n_games', 0)
#     self.epsilon = checkpoint.get('epsilon', 0)
#     self.memoire = checkpoint.get('memoire', deque(maxlen=MAX_MEMOIRE))

# agent.py

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, TAILLE_BLOC
from model import LinQNet, LSTMNet  # Assurez-vous d'importer les deux modèles
import torch.nn as nn

MAX_MEMOIRE = 100_000_000
TAILLE_BATCH = 1000
LR = 0.001

class Agent:
    def __init__(self, model_type, args):
        self.n_games = 0
        self.epsilon = 0  # Paramètre pour l'exploration
        self.gamma = 0.9  # Facteur de discount
        self.memoire = deque(maxlen=MAX_MEMOIRE)
        self.model_type = model_type  # 'linear' ou 'lstm'

        # Initialisation du modèle en fonction du type
        if self.model_type == 'linear':
            self.model = LinQNet(11, 256, 3)
            self.model_file = 'linear_model.pth'
            self.agent_file = 'linear_agent.pth'
        elif self.model_type == 'lstm':
            self.sequence_length = args[0]  # Longueur de la séquence pour LSTM
            self.state_memory = deque(maxlen=self.sequence_length)
            self.model = LSTMNet(11, 256, 3)
            self.model_file = 'lstm_model.pth'
            self.agent_file = 'lstm_agent.pth'
        else:
            raise ValueError("Type de modèle non reconnu. Utilisez 'linear' ou 'lstm'.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def get_state(self, game):
        if self.model_type == 'linear':
            # État pour le modèle linéaire (identique à l'implémentation précédente)
            état = self._get_current_state(game)
            return np.array(état, dtype=int)
        elif self.model_type == 'lstm':
            # Gestion de la séquence pour LSTM
            état = self._get_current_state(game)
            self.state_memory.append(état)
            if len(self.state_memory) < self.sequence_length:
                # Remplir avec des zéros si la séquence n'est pas complète
                padding = [np.zeros_like(état) for _ in range(self.sequence_length - len(self.state_memory))]
                sequence = padding + list(self.state_memory)
            else:
                sequence = list(self.state_memory)
            return np.array(sequence, dtype=int)
        else:
            raise ValueError("Type de modèle non reconnu.")

    def _get_current_state(self, game):
        # Implémentation de l'état courant
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
        # Conversion en tenseurs
        if self.model_type == 'linear':
            états = torch.tensor(np.array(états), dtype=torch.float)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        elif self.model_type == 'lstm':
            états = torch.tensor(np.array(états), dtype=torch.float)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        else:
            raise ValueError("Type de modèle non reconnu.")

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = np.array(dones)

        # Déplacer les tenseurs sur le bon dispositif (CPU ou GPU)
        états = états.to(self.model.lin.weight.device)
        next_states = next_states.to(self.model.lin.weight.device)
        rewards = rewards.to(self.model.lin.weight.device)

        # Prédictions Q-values pour l'état courant
        if self.model_type == 'linear':
            pred = self.model(états)
            target = pred.clone()
            # Calcul de la valeur Q cible
            with torch.no_grad():
                target_next = self.model(next_states)
            for idx in range(len(dones)):
                Q_new = rewards[idx]
                if not dones[idx]:
                    Q_new = rewards[idx] + self.gamma * torch.max(target_next[idx])
                target[idx][actions[idx].item()] = Q_new

            # Optimisation
            self.optimizer.zero_grad()
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

        elif self.model_type == 'lstm':
            # Initialisation de l'état caché
            h0 = torch.zeros(1, états.size(0), self.model.hidden_size).to(états.device)
            c0 = torch.zeros(1, états.size(0), self.model.hidden_size).to(états.device)

            # Prédictions Q-values pour l'état courant
            pred, _ = self.model(états, (h0, c0))
            target = pred.clone()

            # Calcul de la valeur Q cible
            with torch.no_grad():
                next_pred, _ = self.model(next_states, (h0, c0))

            for idx in range(len(dones)):
                Q_new = rewards[idx]
                if not dones[idx]:
                    Q_new = rewards[idx] + self.gamma * torch.max(next_pred[idx])
                target[idx][torch.argmax(actions[idx]).item()] = Q_new


            # Optimisation
            self.optimizer.zero_grad()
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

        else:
            raise ValueError("Type de modèle non reconnu.")

    def get_action(self, état, score):
        # Stratégie d'exploration vs exploitation
        self.epsilon = max(0.01, 0.1 - (self.n_games * 0.001))/(1+score)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            if self.model_type == 'linear':
                état0 = torch.tensor(état, dtype=torch.float)
                état0 = état0.to(self.model.lin.weight.device)
                with torch.no_grad():
                    prediction = self.model(état0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
            elif self.model_type == 'lstm':
                état_sequence = torch.tensor(np.array([état]), dtype=torch.float)  # Ajouter dimension batch
                état_sequence = état_sequence.to(self.model.lin.weight.device)
                with torch.no_grad():
                    h0 = torch.zeros(1, état_sequence.size(0), self.model.hidden_size).to(état_sequence.device)
                    c0 = torch.zeros(1, état_sequence.size(0), self.model.hidden_size).to(état_sequence.device)
                    prediction, _ = self.model(état_sequence, (h0, c0))
                move = torch.argmax(prediction).item()
                final_move[move] = 1
            else:
                raise ValueError("Type de modèle non reconnu.")

        return final_move

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.agent_file
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'model_type': self.model_type
        }
        torch.save(checkpoint, file_name)

    def load(self, file_name=None):
        if file_name is None:
            file_name = self.agent_file
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint.get('n_games', 0)
        self.epsilon = checkpoint.get('epsilon', 0)
        self.model_type = checkpoint.get('model_type', 'linear')

# game.py
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    DROITE = 1
    GAUCHE = 2
    HAUT = 3
    BAS = 4

Point = namedtuple('Point', 'x, y')

# Couleurs
BLANC = (255, 255, 255)
ROUGE = (200, 0, 0)
NOIR = (0, 0, 0)
BLEU = (0, 0, 255)

# Taille du jeu
TAILLE_BLOC = 20
VITESSE = 1000

class SnakeGameAI:

    def __init__(self, largeur=640, hauteur=480):
        self.largeur = largeur
        self.hauteur = hauteur
        # Initialisation de l'écran
        self.display = pygame.display.set_mode((self.largeur, self.hauteur))
        pygame.display.set_caption('Snake IA')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # État initial
        self.direction = Direction.DROITE
        self.tête = Point(self.largeur/2, self.hauteur/2)
        self.serpent = [self.tête,
                        Point(self.tête.x - TAILLE_BLOC, self.tête.y),
                        Point(self.tête.x - (2 * TAILLE_BLOC), self.tête.y)]
        self.score = 0
        self.nourriture = None
        self._place_nourriture()
        self.cumulative_reward = 0
        self.frame_iteration = 0

    def _place_nourriture(self):
        x = random.randint(0, (self.largeur - TAILLE_BLOC) // TAILLE_BLOC) * TAILLE_BLOC
        y = random.randint(0, (self.hauteur - TAILLE_BLOC) // TAILLE_BLOC) * TAILLE_BLOC
        self.nourriture = Point(x, y)
        if self.nourriture in self.serpent:
            self._place_nourriture()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Déplacement
        self._move(action)  # Met à jour la tête
        self.serpent.insert(0, self.tête)

        # 3. Vérifier la collision
        # reward = 0
        # if self.score>0:
        #     reward = -0.001
        #     # reward = -0.001*(abs(self.nourriture.x-self.tête.x)/TAILLE_BLOC+abs(self.nourriture.y-self.tête.y)/TAILLE_BLOC - 10)
        
        reward = -0.001
        game_over = False
        if self._collision() or self.frame_iteration > 100 * len(self.serpent):
            game_over = True
            reward += -10
            return reward, game_over, self.score

        # 4. Vérifier si le serpent a mangé la nourriture
        if self.tête == self.nourriture:
            self.score += 1
            reward += 20
            self._place_nourriture()
        else:
            self.serpent.pop()

        # 5. Mettre à jour l'UI et l'horloge
        self._update_ui()
        self.clock.tick(VITESSE)

        self.cumulative_reward += reward
        # 6. Retourner le jeu_over et le score
        return reward, game_over, self.score

    def _collision(self, pt=None):
        if pt is None:
            pt = self.tête
        # Vérifier les limites
        if pt.x > self.largeur - TAILLE_BLOC or pt.x < 0 or pt.y > self.hauteur - TAILLE_BLOC or pt.y < 0:
            return True
        # Vérifier la collision avec le corps
        if pt in self.serpent[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(NOIR)
        for pt in self.serpent:
            pygame.draw.rect(self.display, BLEU, pygame.Rect(pt.x, pt.y, TAILLE_BLOC, TAILLE_BLOC))
            pygame.draw.rect(self.display, BLEU, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, ROUGE, pygame.Rect(self.nourriture.x, self.nourriture.y, TAILLE_BLOC, TAILLE_BLOC))

        text = font.render("Score: " + str(self.score), True, BLANC)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # def _move(self, action):
    #     # [tout droit, droite, gauche]

    #     hor_dir = [Direction.GAUCHE, Direction.DROITE]
    #     ver_dir = [Direction.HAUT, Direction.BAS]

    #     index_dir = hor_dir.index(self.direction) if self.direction in hor_dir else ver_dir.index(self.direction)

    #     if action[1] == 1:  # Tourner à droite
    #         index_dir = (index_dir + 1) % 2
    #         self.direction = hor_dir[index_dir] if self.direction in hor_dir else ver_dir[index_dir]
    #     elif action[2] == 1:  # Tourner à gauche
    #         index_dir = (index_dir - 1) % 2
    #         self.direction = hor_dir[index_dir] if self.direction in hor_dir else ver_dir[index_dir]
    #     # Sinon, continuer tout droit

    #     x = self.tête.x
    #     y = self.tête.y
    #     if self.direction == Direction.DROITE:
    #         x += TAILLE_BLOC
    #     elif self.direction == Direction.GAUCHE:
    #         x -= TAILLE_BLOC
    #     elif self.direction == Direction.HAUT:
    #         y -= TAILLE_BLOC
    #     elif self.direction == Direction.BAS:
    #         y += TAILLE_BLOC

    #     self.tête = Point(x, y)
    
    def _move(self, action):
        # [tout droit, droite, gauche]

        clock_wise = [Direction.DROITE, Direction.BAS, Direction.GAUCHE, Direction.HAUT]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # Aller tout droit
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Tourner à droite
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Tourner à gauche
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.tête.x
        y = self.tête.y
        if self.direction == Direction.DROITE:
            x += TAILLE_BLOC
        elif self.direction == Direction.GAUCHE:
            x -= TAILLE_BLOC
        elif self.direction == Direction.BAS:
            y += TAILLE_BLOC
        elif self.direction == Direction.HAUT:
            y -= TAILLE_BLOC

        self.tête = Point(x, y)


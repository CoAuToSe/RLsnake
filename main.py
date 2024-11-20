# main.py
from agent import Agent
from game import SnakeGameAI
import os
import numpy as np
# import matplotlib.pyplot as plt
# from IPython import display

# plt.ion()

# def plot(scores, mean_scores):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.clf()
#     plt.title('Entraînement...')
#     plt.xlabel('Nombre de parties')
#     plt.ylabel('Score')
#     plt.plot(scores)
#     plt.plot(mean_scores)
#     plt.ylim(ymin=0)
#     plt.text(len(scores)-1, scores[-1], str(scores[-1]))
#     plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
#     plt.show(block=False)
#     plt.pause(.1)

import matplotlib.pyplot as plt

plt.ion()  # Active le mode interactif

def plot(scores, mean_scores):
    # fig = plt.figure(1)
    # ax = plt.gca()
    # ax.set_facecolor("xkcd:purple")
    # fig.patch.set_facecolor("xkcd:grey")
    # plt.clf()
    # plt.title('Entraînement...')
    # plt.xlabel('Nombre de parties')
    # plt.ylabel('Score')
    # plt.plot(scores, label='Score')
    # plt.plot(mean_scores, label='Score Moyen')
    # plt.ylim(ymin=0)
    # plt.legend()
    # plt.draw()
    # plt.pause(0.1)
    pass
    
def save_scores(scores, mean_scores, file_name):
    np.savez(file_name, scores=scores, mean_scores=mean_scores)

def load_scores(file_name):
    if os.path.exists(file_name):
        data = np.load(file_name)
        return list(data['scores']), list(data['mean_scores'])
    else:
        return [], []


def train(model_type):
    # scores = []
    # mean_scores = []
    # total_score = 0
    # record = 0
    agent = Agent(model_type)

    # Sélection des noms de fichiers en fonction du modèle
    if model_type == 'linear':
        agent_model_file = 'linear_agent.pth'
        score_file = 'linear_scores.npz'
    elif model_type == 'lstm':
        agent_model_file = 'lstm_agent.pth'
        score_file = 'lstm_scores.npz'
    else:
        raise ValueError("Type de modèle non reconnu. Utilisez 'linear' ou 'lstm'.")

    # Charger l'agent s'il existe un fichier de sauvegarde
    if os.path.exists(agent_model_file):
        agent.load(agent_model_file)
        print(f"Agent chargé avec succès depuis {agent_model_file}. Nombre de parties précédentes : {agent.n_games}")

    scores, mean_scores = load_scores(score_file)
    total_score = sum(scores)
    record = max(scores) if scores else 0
    
    game = SnakeGameAI()
    try:
        while True:
            # Récupérer l'état actuel
            état_old = agent.get_state(game)

            # Obtenir le mouvement
            action = agent.get_action(état_old)

            # Effectuer le mouvement et obtenir le nouvel état
            reward, done, score = game.play_step(action)
            état_new = agent.get_state(game)

            # Entraîner la mémoire courte
            # agent.train_step(état_old, action, reward, état_new, done)
            
            # Entraîner la mémoire courte
            agent.train_step([état_old], [action], [reward], [état_new], [done])


            # Mémoriser l'expérience
            agent.remember(état_old, action, reward, état_new, done)

            if done:
                # Entraînement à long terme
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # if score > record:
                #     record = score
                #     # agent.model.save()
                
                        
                if score > record:
                    record = score
                    agent.save('best_'+agent_model_file)  # Sauvegarder le meilleur modèle

                # Sauvegarder l'agent après chaque partie
                agent.save(agent_model_file)

                print('Partie', agent.n_games, 'Score', score, 'Record:', record)

                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)
                plot(scores, mean_scores)
                save_scores(scores, mean_scores, score_file)
        # plt.ioff()
        # plt.show()
    except KeyboardInterrupt:
        print("Interruption détectée. Sauvegarde de l'agent...")
        agent.save('interupt_'+agent_model_file)
        save_scores(scores, mean_scores, score_file)
        print("Agent sauvegardé. Fermeture du programme.")
    
if __name__ == '__main__':
    train('lstm')

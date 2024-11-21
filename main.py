# main.py
from agent import Agent
from ppo_agent import PPOAgent
from game import SnakeGameAI
import os
import numpy as np
import torch
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

def plot(scores, mean_scores, rewards):
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
    # plt.plot(rewards, label='Reward')
    # plt.ylim(ymin=0)
    # plt.legend()
    # plt.draw()
    # plt.pause(0.1)
    pass
    
def save_scores(scores, mean_scores, rewards, file_name):
    np.savez(file_name, scores=scores, mean_scores=mean_scores, rewards=rewards)

def load_scores(file_name):
    if os.path.exists(file_name):
        data = np.load(file_name)
        return list(data['scores']), list(data['mean_scores']), list(data['rewards'])
    else:
        return [], [], []

def save_scores_ppo(scores, filename):
    np.savez(filename, scores=scores)

def load_scores_ppo(filename):
    if os.path.exists(filename):
        data = np.load(filename)
        return list(data['scores'])
    else:
        return []

def train_ppo():
    input_size = 11  # Selon votre état
    hidden_size = 256
    output_size = 3  # Nombre d'actions possibles
    agent = PPOAgent(input_size, hidden_size, output_size)
    game = SnakeGameAI()
    max_episodes = 100_000
    
    agent_model_file = 'ppo_agent.pth'
    score_file = 'ppo_scores.npz'
    
    scores = load_scores_ppo(score_file)
    best_score = 0
    
    if os.path.exists(agent_model_file):
        agent.load(agent_model_file)

    try:
        for episode in range(agent.episode, max_episodes):
            agent.episode = episode
            game.reset()
            state = agent.get_state(game) 
            done = False
            while not done:
                action, action_logprob = agent.select_action(state)
                final_move = [0, 0, 0]
                final_move[action] = 1
                reward, done, score = game.play_step(final_move)
                next_state = agent.get_state(game)
                _, value = agent.model(torch.FloatTensor(state))
                agent.store_transition(state, action, reward, next_state, done, action_logprob, value.item())
                state = next_state
                if done:
                    agent.train()
                    print(f"Épisode: {episode}, Score: {score}, Best Score: {best_score}")
                    if score > best_score:
                        best_score = score
                        agent.save('best_'+agent_model_file)

                    # Sauvegarder l'agent après chaque épisode
                    agent.save(agent_model_file)
                    scores.append(score)
                    save_scores_ppo(scores, score_file)
    except KeyboardInterrupt:
        print("Interruption détectée. Sauvegarde de l'agent...")
        agent.save(agent_model_file)
        save_scores_ppo(scores, score_file)
        print("Agent sauvegardé. Fermeture du programme.")

def train(model_type, args = []):
    # scores = []
    # mean_scores = []
    # total_score = 0
    # record = 0
    agent = Agent(model_type, args)

    # Sélection des noms de fichiers en fonction du modèle
    if model_type == 'linear':
        agent_model_file = 'linear_agent.pth'
        score_file = 'linear_scores.npz'
    elif model_type == 'lstm':
        len_mem = args[0]
        agent_model_file = f"lstm_{len_mem}_agent.pth"
        score_file = f"lstm_{len_mem}_scores.npz"
    else:
        raise ValueError("Type de modèle non reconnu. Utilisez 'linear' ou 'lstm'.")

    # Charger l'agent s'il existe un fichier de sauvegarde
    if os.path.exists(agent_model_file):
        agent.load(agent_model_file)
        print(f"Agent chargé avec succès depuis {agent_model_file}. Nombre de parties précédentes : {agent.n_games}")

    scores, mean_scores, rewards = load_scores(score_file)
    score = 0
    total_score = sum(scores)
    record = max(scores) if scores else 0
    
    game = SnakeGameAI()
    try:
        while True:
            # Récupérer l'état actuel
            état_old = agent.get_state(game)

            # Obtenir le mouvement
            action = agent.get_action(état_old, score)

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
                cumul_reward = game.cumulative_reward
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

                print('Partie', agent.n_games, 'Score', score, 'Record:', record, 'Reward',cumul_reward)

                scores.append(score)
                rewards.append(cumul_reward)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)
                plot(scores, mean_scores, rewards)
                save_scores(scores, mean_scores, rewards, score_file)
        # plt.ioff()
        # plt.show()
    except KeyboardInterrupt:
        print("Interruption détectée. Sauvegarde de l'agent...")
        agent.save('interupt_'+agent_model_file)
        save_scores(scores, mean_scores, rewards, score_file)
        print("Agent sauvegardé. Fermeture du programme.")
    
if __name__ == '__main__':
    # train('lstm', [100])
    train_ppo()

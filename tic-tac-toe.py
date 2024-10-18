import numpy as np
import random

# Параметры обучения
alpha = 0.1  # скорость обучения
gamma = 0.9  # коэффициент дисконтирования
epsilon = 0.1  # вероятность исследования

# Таблица Q-значений
Q = np.zeros((3**9, 9))

# Функция для получения состояния из доски
def get_state(board):
    state = 0
    for i in range(9):
        if board[i] == 'X':
            state += 1 * (3**i)
        elif board[i] == 'O':
            state += 2 * (3**i)
    return state

# Функция для выбора действия
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 8)
    else:
        return np.argmax(Q[state])

# Функция для обновления Q-значений
def update_Q(state, action, reward, next_state):
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

# Функция для игры
def play_game(agent, opponent):
    board = [' '] * 9
    current_player = 'X'
    while not check_win(board, current_player) and ' ' in board:
        if current_player == 'X':
            state = get_state(board)
            action = agent.choose_action(state, epsilon)
            board[action] = 'X'
        else:
            action = opponent.choose_action(board)
            board[action] = 'O'
        current_player = 'O' if current_player == 'X' else 'X'

    # Награждение агента
    reward = 1 if check_win(board, 'X') else -1 if check_win(board, 'O') else 0
    return reward

# Функция для проверки победы
def check_win(board, player):
    win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                       (0, 3, 6), (1, 4, 7), (2, 5, 8),
                       (0, 4, 8), (2, 4, 6)]
    for combination in win_combinations:
        if board[combination[0]] == board[combination[1]] == board[combination[2]] == player:
            return True
    return False

# Определение агента и оппонента
class Agent:
    def choose_action(self, state, epsilon):
        return choose_action(state, epsilon)

class RandomOpponent:
    def choose_action(self, board):
        available_moves = [i for i in range(9) if board[i] == ' ']
        return random.choice(available_moves)

# Инициализация агента и оппонента
agent = Agent()
opponent = RandomOpponent()

# Обучение агента
rewards = []
for i in range(10000):
    reward = play_game(agent, opponent)
    rewards.append(reward)

# Построение кривой обучения
average_rewards = [np.mean(rewards[i-100:i]) for i in range(100, len(rewards))]

# Построение графика
import matplotlib.pyplot as plt
plt.plot(average_rewards)
plt.xlabel("Количество шагов обучения")
plt.ylabel("Средняя награда")
plt.show()
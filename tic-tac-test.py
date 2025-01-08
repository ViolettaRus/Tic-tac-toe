import numpy as np
import random
import matplotlib.pyplot as plt

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
def choose_action(state, epsilon, Q, board):
    possible_actions = [i for i, val in enumerate(board) if val == ' ']
    if random.random() < epsilon:  # Исследование
        return random.choice(possible_actions)
    else:  # Эксплуатация
        q_values = [Q[state, action] if action in possible_actions else -np.inf for action in range(9)]
        return np.argmax(q_values)

# Функция для обновления Q-значений
def update_Q(state, action, reward, next_state, Q):
    if next_state is None:  # Конечное состояние
        Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])
    else:  # Обычное состояние
        max_next_q = max(Q[next_state, a] for a in range(9))
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * max_next_q - Q[state, action])

# Функция для проверки победы
def check_win(board):
    win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                        (0, 3, 6), (1, 4, 7), (2, 5, 8),
                        (0, 4, 8), (2, 4, 6)]
    for combination in win_combinations:
        if board[combination[0]] == board[combination[1]] == board[combination[2]] != ' ':
            return board[combination[0]]
    return None

# Функция для игры
def play_game(agent, opponent):
    board = [' '] * 9
    current_player = 'X'
    state = get_state(board)
    total_reward = 0

    while True:
        if current_player == 'X':  # Ход агента
            action = agent.choose_action(state, epsilon, Q, board)
        else:  # Ход оппонента
            action = opponent.choose_action(board)

        if board[action] != ' ':
            # Недопустимый ход
            reward = -10 if current_player == 'X' else 0
            next_state = None
            update_Q(state, action, reward, next_state, Q)
            return reward

        board[action] = current_player
        winner = check_win(board)

        if winner:  # Если есть победитель
            reward = 1 if winner == 'X' else -1
            next_state = None
            update_Q(state, action, reward, next_state, Q)
            return reward

        if ' ' not in board:  # Ничья
            reward = 0.5
            next_state = None
            update_Q(state, action, reward, next_state, Q)
            return reward

        next_state = get_state(board)
        if current_player == 'X':  # Обновляем Q для агента
            update_Q(state, action, 0, next_state, Q)

        state = next_state
        current_player = 'O' if current_player == 'X' else 'X'

# Определение агента и оппонента
class Agent:
    def choose_action(self, state, epsilon, Q, board):
        return choose_action(state, epsilon, Q, board)

class RandomOpponent:
    def choose_action(self, board):
        available_moves = [i for i in range(9) if board[i] == ' ']
        return random.choice(available_moves)

# Инициализация агента и оппонента
agent = Agent()
opponent = RandomOpponent()

# Обучение агента
rewards = []
episodes = 20000

for _ in range(episodes):
    reward = play_game(agent, opponent)
    rewards.append(reward)

# Построение кривой обучения (скользящее среднее)
window_size = 100
average_rewards = np.convolve(rewards, np.ones(window_size), 'valid') / window_size

# Построение графика
plt.plot(average_rewards)
plt.xlabel("Количество игр")
plt.ylabel("Средняя награда")
plt.title("Кривая обучения агента")
plt.show()
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

env = gym.make("CartPole-v1")

def create_q_network():
    """Creates a neural network for Q-value approximation."""
    net_input = Input(shape=(4,))
    x = Dense(64, activation='relu')(net_input)
    x = Dense(32, activation='relu')(x)
    output = Dense(2, activation='linear')(x)  
    return Model(inputs=net_input, outputs=output)

q_net = create_q_network()
q_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

num_episodes = 500
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.995  
epsilon_min = 0.01  

episode_rewards = []

def visualize_cartpole(state):
    """Visualizes the CartPole environment."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    cart_width, cart_height = 100, 20
    cart_x = int(state[0] * 100 + 300)  
    cart_y = 350

    cv2.rectangle(img, (cart_x - cart_width // 2, cart_y - cart_height // 2),
                  (cart_x + cart_width // 2, cart_y + cart_height // 2), (0, 0, 255), -1)

    pole_length = 100
    pole_angle = state[2]
    pole_end_x = int(cart_x + pole_length * np.sin(pole_angle))
    pole_end_y = int(cart_y - pole_length * np.cos(pole_angle))
    cv2.line(img, (cart_x, cart_y), (pole_end_x, pole_end_y), (255, 0, 0), 5)

    cv2.line(img, (0, 380), (600, 380), (0, 255, 0), 5)

    cv2.imshow("CartPole", img)
    cv2.waitKey(1)

def epsilon_greedy_policy(state, epsilon):
    """Selects an action using the epsilon-greedy strategy."""
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  
    q_values = q_net.predict(state.reshape(1, -1), verbose=0)  
    return np.argmax(q_values[0])  

for episode in range(num_episodes):
    state, _ = env.reset()  
    done = False
    total_reward = 0

    while not done:
        visualize_cartpole(state)  

        action = epsilon_greedy_policy(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)

        q_values = q_net.predict(state.reshape(1, -1), verbose=0)[0]
        next_q_values = q_net.predict(next_state.reshape(1, -1), verbose=0)[0]

        target = reward + gamma * np.max(next_q_values) * (1 - int(terminated))
        q_values[action] = target 

        q_net.fit(state.reshape(1, -1), q_values.reshape(1, -1), epochs=1, verbose=0)

        state = next_state
        total_reward += reward

        done = terminated or truncated

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

cv2.destroyAllWindows()

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title("Total Rewards per Episode (Semi-Gradient Q-Learning)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()

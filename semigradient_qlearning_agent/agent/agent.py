import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

# Initialize the CartPole environment
env = gym.make("CartPole-v1")

def create_q_network():
    """Creates a neural network for Q-value approximation."""
    net_input = Input(shape=(4,))
    x = Dense(64, activation='relu')(net_input)
    x = Dense(32, activation='relu')(x)
    output = Dense(2, activation='linear')(x)  # Output: Q-values for each action
    return Model(inputs=net_input, outputs=output)

# Create and compile the Q-network
q_net = create_q_network()
q_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Hyperparameters
num_episodes = 500
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum value of epsilon

# For tracking rewards per episode
episode_rewards = []

def visualize_cartpole(state):
    """Visualizes the CartPole environment."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Cart parameters
    cart_width, cart_height = 100, 20
    cart_x = int(state[0] * 100 + 300)  # Scale and shift cart position
    cart_y = 350

    # Draw cart
    cv2.rectangle(img, (cart_x - cart_width // 2, cart_y - cart_height // 2),
                  (cart_x + cart_width // 2, cart_y + cart_height // 2), (0, 0, 255), -1)

    # Draw pole
    pole_length = 100
    pole_angle = state[2]
    pole_end_x = int(cart_x + pole_length * np.sin(pole_angle))
    pole_end_y = int(cart_y - pole_length * np.cos(pole_angle))
    cv2.line(img, (cart_x, cart_y), (pole_end_x, pole_end_y), (255, 0, 0), 5)

    # Draw ground line
    cv2.line(img, (0, 380), (600, 380), (0, 255, 0), 5)

    # Show the visualization
    cv2.imshow("CartPole", img)
    cv2.waitKey(1)

def epsilon_greedy_policy(state, epsilon):
    """Selects an action using the epsilon-greedy strategy."""
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Random action
    q_values = q_net.predict(state.reshape(1, -1), verbose=0)  # Predict Q-values
    return np.argmax(q_values[0])  # Choose action with the highest Q-value

for episode in range(num_episodes):
    state, _ = env.reset()  # Reset the environment and get the initial state
    done = False
    total_reward = 0

    while not done:
        visualize_cartpole(state)  # Visualize the environment

        # Choose an action using the epsilon-greedy policy
        action = epsilon_greedy_policy(state, epsilon)

        # Take the action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Predict Q-values for the current and next state
        q_values = q_net.predict(state.reshape(1, -1), verbose=0)[0]
        next_q_values = q_net.predict(next_state.reshape(1, -1), verbose=0)[0]

        # Compute the target value using the semi-gradient Q-learning update
        target = reward + gamma * np.max(next_q_values) * (1 - int(terminated))
        q_values[action] = target  # Update only the Q-value for the taken action

        # Train the Q-network to minimize the MSE between prediction and target
        q_net.fit(state.reshape(1, -1), q_values.reshape(1, -1), epochs=1, verbose=0)

        # Update the state and accumulate the reward
        state = next_state
        total_reward += reward

        # Check if the episode is done
        done = terminated or truncated

    # Store the total reward for this episode
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Decay epsilon to reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Close the OpenCV windows
cv2.destroyAllWindows()

# Plot the total rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title("Total Rewards per Episode (Semi-Gradient Q-Learning)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()

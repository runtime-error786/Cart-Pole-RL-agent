import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

num_episodes = 500  
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

for episode in range(num_episodes):
    state, _ = env.reset()  
    done = False
    total_reward = 0

    while not done:
        visualize_cartpole(state)  

        action = env.action_space.sample()  
        next_state, reward, terminated, truncated, info = env.step(action)  

        total_reward += reward 
        state = next_state  

        done = terminated or truncated 

    episode_rewards.append(total_reward)  
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

cv2.destroyAllWindows()

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title("Total Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()

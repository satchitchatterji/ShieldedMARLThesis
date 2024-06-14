import gym
import gym_safety
env = gym.make('GridNav-v0', gridsize=5)
# env = gym.make('CartSafe-v0')

obs = env.reset() # difference: no info
num_episodes = 10

print(env.observation_space)
print(env.action_space)
print(env.obstacle_states)
exit()
print(obs)
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:

        env.render()
        
        # Select a random action (either 0 or 1)
        action = env.action_space.sample()
        
        # Take the action and observe the result
        next_obs, reward, done, info = env.step(action)
        
        # Update total reward for this episode
        total_reward += reward
        
        # Update the current obs
        obs = next_obs
        
        if done:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            break

# Close the environment
env.close()
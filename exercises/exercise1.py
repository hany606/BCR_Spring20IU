import gym


env = gym.make("CartPole-v0")

init_obs = env.reset()

num_episodes = 1

def get_action(obs):
    return env.action_space.sample()

def run_episode(render=False):
    done  = False
    obs = env.reset
    while not done:
        if(render):
            env.render()
        action = get_action(obs)
        obs, reward, done, _ = env.step(action)



run_episode(render=True)

env.close()
    
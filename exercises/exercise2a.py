import gym
import numpy as np

env = gym.make('CartPole-v0')

num_episodes = 100
num_steps = 200
num_iterations = 100
train_params = {"EHC": {"mutation_step_size": 0.02, "population_size": 20}}


# NN params
pvariance = 0.1
ppvariance = 0.02
nhiddens = 5

ninputs = env.observation_space.shape[0]
if(isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

num_HP_NN = nhiddens*(ninputs+noutputs + 1) + noutputs 



W1 = np.random.randn(nhiddens,ninputs) * pvariance
W2 = np.random.randn(noutputs, nhiddens) * pvariance
b1 = np.zeros(shape=(nhiddens, 1))
b2 = np.zeros(shape=(noutputs, 1))

NN_init_params = {'W1': W1, 'W2': W2,
                  'b1': b1, 'b2': b2}


def random_action_policy():
    return env.action_space.sample()

def NN_policy(observation, NN_params):
    W1 = NN_params['W1']
    W2 = NN_params['W2']
    b1 = NN_params['b1']
    b2 = NN_params['b2']
    
    observation.resize(ninputs, 1)
    Z1 = np.dot(W1, observation)
    Z1 += b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)
    Z2 += b2
    A2 = np.tanh(Z2)
    if (isinstance(env.action_space, gym.spaces.box.Box)):
        action = A2
    else:
        action = np.argmax(A2)
    return action

def get_action(observation, other):
    # return random_sample_action()
    return NN_policy(observation, other['NN_params'])
    
def run_episode(NN_params = None, render=False):
    done  = False
    obs = env.reset()
    cumulative_reward = 0
    while not done:
        if(render):
            env.render()

        if(NN_params is None):
            action = random_action_policy()
        else:
            action = get_action(obs, {'NN_params': NN_params})
        obs, reward, done, _ = env.step(action)
        cumulative_reward += reward
    return cumulative_reward


def evolutionary_hill_climber(params):
    
    def evaluate(params):
        cumulative_reward = 0
        for i in range(num_episodes):
            cumulative_reward += run_episode(NN_params=params)
        return cumulative_reward
    
    def rank(scores):
        # print(scores)
        ranks = np.argsort(scores)[::-1]
        # print(ranks)
        # for i in range(len(scores)):
        #     print("score: {:}".format(scores[ranks[i]]))
        #     # print("score: {:}, rank: {:}".format(scores[i], ranks[i]))
        # exit()
        return ranks

    def sample_noise():
        return np.random.randn(1,num_HP_NN) * ppvariance

    def mutate(theta, noise):
        return theta + noise

    def theta_to_params(theta):
        w1_shape = NN_init_params["W1"].shape
        w2_shape = NN_init_params["W2"].shape
        b1_shape = NN_init_params["b1"].shape
        b2_shape = NN_init_params["b2"].shape
        return {"W1": np.array(theta[:w1_shape[0]*w1_shape[1]]).reshape(w1_shape),
                "W2": np.array(theta[w1_shape[0]*w1_shape[1]:w1_shape[0]*w1_shape[1]+w2_shape[0]*w2_shape[1]]).reshape(w2_shape),
                "b1": np.array(theta[w1_shape[0]*w1_shape[1]+w2_shape[0]*w2_shape[1]:w1_shape[0]*w1_shape[1]+w2_shape[0]*w2_shape[1]+b1_shape[0]*b1_shape[1]]).reshape(b1_shape),
                "b2": np.array(theta[w1_shape[0]*w1_shape[1]+w2_shape[0]*w2_shape[1]+b1_shape[0]*b1_shape[1]:w1_shape[0]*w1_shape[1]+w2_shape[0]*w2_shape[1]+b1_shape[0]*b1_shape[1]+b2_shape[0]*b2_shape[1]]).reshape(b2_shape),
        }
    theta_params = np.random.randn(params["EHC"]["population_size"],num_HP_NN)*pvariance

    for g in range(num_iterations):
        print("#{:}th iteration:".format(g))
        # print(theta_params)
        print("----------------------")
        population_evaluation_scores = []
        for i in range(params["EHC"]["population_size"]):
            training_params = theta_to_params(theta_params[i])
            population_evaluation_scores.append(evaluate(training_params))
            # Rank the individuals in u
        u = rank(population_evaluation_scores)
        for l in range(params["EHC"]["population_size"]//2):
            noise = sample_noise()
            theta_params[u[l+params["EHC"]["population_size"]//2]] = mutate(theta_params[u[l]], noise)
    
    # print(theta_params[0])
    cumulative_reward = 0
    for i in range(num_episodes):
        reward = run_episode(NN_params=theta_to_params(theta_params[0]), render=True)
        print(reward)
        cumulative_reward += reward
    print(cumulative_reward)
if __name__ == '__main__':
    # for i in range(num_episodes):
        # run_episode(NN_params= NN_init_params, render=True)
    evolutionary_hill_climber(train_params)

    env.close()
    

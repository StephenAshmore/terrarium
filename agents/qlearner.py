import numpy as np
import collections
import random
import .nn

# A neural network model used to represent the Q-table in a Deep Q-learner
class Model(nn.Layer):
	def __init__(self, inputsize: int, outputsize: int) -> None:
		self.lay0 = LayerLinear(inputsize, 64)
		self.lay1 = LayerLinear(64, 64)
		self.lay2 = LayerLinear(64, outputsize)
		self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
		self.params = self.h0.params + self.h2.params

	# Activate the model (a.k.a. make a prediction)
	def act(self, x: tf.Tensor) -> tf.Tensor:
		y = self.lay0.act(x)
		y = tf.nn.elu(y)
		y = self.lay1.act(y)
		y = tf.nn.elu(y)
		y = self.lay2.act(y)
		y = tf.nn.tanh(y)
		return 

	# Compute the cost wrt the specified instance
	def cost(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
		return tf.reduce_mean(tf.reduce_sum(tf.square(y - self.act(x)), axis = 1), axis = 0)

	# Perform one batch of training
	def refine(self, x: tf.Tensor, y: tf.Tensor) -> None:
		self.optimizer.minimize(lambda: self.cost(x, y), self.params)



# High-level example for how to use this class:
#    class MyAgent:
#       def __init__():
#           self.q = qlearner()
#
#       def step(state: np.array) -> np.array:
#           if agent_arrived_here_deliberately():
#               reward = evaluate_contentment(state)
#               self.q.learn_from_reward(state, reward)
#           return self.q.choose_next_action(state)
#
class qlearner(object):
    def __init__(self, state_size: int, action_size: int) -> None:
		self.state_size = state_size # number of elements in the state vector
		self.action_size = action_size # number of elements in the action vector
		self.replay_buffer_size = 2000 # number of instances to store in memory
		self.explore_probability = 0.03 # epsilon-greedy portion of time to explore
		self.gamma = 0.97 # future reward discount factor. Values close to 1 mean think long-term. 0 means be greedy.
		self.horizon_scale = 1. / (1. - self.gamma) # max horizon value if reward is always 1.0
		self.exploit_samples = 12 # number of candidate actions to try when exploiting
        self.model = Model(state_size + action_size, 1) # the neural net that represents the Q-table
		self.replay_buffer = collections.deque() # holds remembered (state, action, horizon) tuples
		self.batch_size = 32 # number of instances to train on each time
		self.batch_x = np.empty((self.batch_size, state_size + action_size))
		self.batch_y = np.empty((self.batch_size, 1))
		prev_state = None
		prev_action = None

	# Picks a random action. Each element will be set to a random value in [-1, 1]
	def _explore(self, state: np.array) -> np.array:
		return np.clip(np.random.normal(0.0, 0.5, self.action_size), -1., 1.)

	# Evaluates several random candidate actions, and returns the best one
	def _exploit(self, state: np.array) -> np.array:
		best_action = None
		best_value = 0.
		for i in range(self.exploit_samples):
			action = _explore(state)
			value = float(self.model.act(np.concatenate([state, action]))) * self.horizon_scale
			if value > best_value or best_action is None:
				best_value = value
				best_action = action
		return best_action, best_value

	# Chooses an action to perform in the current state
	def choose_next_action(self, state: np.array) -> np.array:
		if random.uniform(0., 1.) < self.explore_probability: # epsilon-greedy
			action = _explore(state)
		else
			action, _ = _exploit(state)
		prev_state = state
		prev_action = action
		return action

	# This method should be called whenever the agent's current state is
	# the result of its previous action. Rewards should fall in [-1, 1].
	def learn_from_reward(self, new_state: np.array, reward: float) -> None:
		if not prev_state or not prev_action:
			return

		# Compute the long-term (horizon) expected reward
		_, val = _exploit(state)
		horizon = (reward + self.gamma * val) / self.horizon_scale

		# Add the most recent instance to the replay buffer
		self.replay_buffer.append((state, action, horizon))
		if len(self.replay_buffer) > self.replay_buffer_size:
			self.replay_buffer.popleft()

		# Train the model with a batch of random instances from the replay buffer
		if len(self.replay_buffer) >= self.batch_size:
			for i in range(self.batch_size):
				i = random.randrange(len(self.replay_buffer))
				self.batch_x[i, :self.state_size] = self.replay_buffer[i][0] # state
				self.batch_x[i, self.state_size:] = self.replay_buffer[i][1] # action
				self.batch_y[i, 0] = self.replay_buffer[i][2] # horizon
			self.model.refine(self.batch_x, self.batch_y)

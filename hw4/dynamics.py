import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        #self.env= env
        self.n_layers=n_layers
        self.size=size
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization
        self.activation= activation
        self.output_activation = output_activation
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        #if len(env.observation_space.shape) == 1:
        #    input_shape = env.observation_space.shape
        #else:
        #    img_h, img_w, img_c = env.observation_space.shape
        #    input_shape = (img_h, img_w, frame_history_len * img_c)
        #self.num_actions = env.action_space.n
        self.a_dim = env.action_space.shape[0]
        self.ob_dim = env.observation_space.shape[0]
        #self.next_ob_dim = self.ob_dim
        #self.ob = tf.placeholder(tf.float32, shape=[None,self.ob_dim])
        self.s_a = tf.placeholder(tf.float32,shape = [None,self.ob_dim+self.a_dim])
        self.delta = tf.placeholder(tf.float32,shape=[None,self.ob_dim])
        self.delta_predict = build_mlp(self.s_a,self.ob_dim,'predict',self.n_layers,self.size,self.activation,self.output_activation)
        self.loss = tf.losses.mean_squared_error(self.delta,self.delta_predict)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """

        ob = np.concatenate([path["observations"] for path in data])
        ne_ob = np.concatenate([path["next_observations"] for path in data])
        ac_na = np.concatenate([path["actions"] for path in data])
        ob_norm = (ob- self.mean_obs)/(self.std_obs+1e-7)
        a_norm = (ac_na - self.mean_action)/(self.std_action+1e-7)
        delta_norm = (ne_ob-ob- self.mean_deltas)/(self.std_deltas+1e-7)
        s_a = np.concatenate((ob_norm,a_norm),axis=1)
        N = ob.shape[0]
        idx = np.arange(N)
        for _ in range(self.iterations):
            np.random.shuffle(idx)
            for i in range(int(np.ceil(N/self.batch_size))):
                start_idx = i*self.batch_size%N
                new_idx=idx[start_idx:start_idx+self.batch_size]
                s_a_batch = s_a[new_idx]
                #s_a_batch = np.concatenate([a_norm[new_idx],ob_norm],1)
                delta_batch = delta_norm[new_idx] 
                self.sess.run([self.train_op],feed_dict={self.delta:delta_batch,self.s_a:s_a_batch})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        state_norm = (states-self.mean_obs)/(self.std_obs+1e-7)
        action_norm = (actions- self.mean_action)/(self.std_action+1e-7)
        s_a = np.concatenate((state_norm,action_norm),axis=1)
        delta_predict = self.sess.run(self.delta_predict,feed_dict ={self.s_a:s_a})

        return delta_predict*self.std_deltas+self.mean_deltas

        

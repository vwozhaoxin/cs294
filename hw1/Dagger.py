import numpy as np 
import tensorflow as tf 
import keras 
import run_expert
import tf_util
import gym
import load_policy
import math
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import seaborn as sns
import tqdm

class Config(object):

	n_feature= 11
	n_classes = 3
	dropout=0.5
	hidden_size1= 128
	hidden_size2= 256
	hidden_size3 = 64
	batch_size = 256
	n_epochs = 10
	lr = 0.0005
	itera = 20
	train_itera = 20
	envname = 'Hopper-v1'
	max_steps = 1000

class Model(object):

	def __init__(self,config):
		self.config= config
		self.build()

	def add_placerholders(self):
		self.input_placeholder =tf.placeholder(tf.float32,shape=(None,self.config.n_feature))
		self.labels_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.n_classes))
		self.training_placeholder = tf.placeholder(tf.bool)
		self.dropout_placeholder = tf.placeholder(tf.float32)


	def create_feed_dict(self,input_batch,labels_batch=None,dropout=1,is_training=False):
		feed_dict ={self.input_placeholder:input_batch,
					self.dropout_placeholder:dropout,
					self.training_placeholder:is_training}
		if labels_batch is not None:
			feed_dict[self.labels_placeholder]=labels_batch
		return feed_dict

	def prediction_op(self):
		x=self.input_placeholder
		layer1 = tf.layers.dense(x,self.config.hidden_size1,activation=tf.nn.relu)
		#layer2 = tf.nn.dropout(layer1,keep_prob=self.dropout_placeholder)  no need dropout
		layer3 = tf.layers.dense(layer1,self.config.hidden_size2,activation=tf.nn.relu)
		#layer4 =tf.nn.dropout(layer3,keep_prob=self.dropout_placeholder)
		layer5 = tf.layers.dense(layer3,self.config.hidden_size3,activation=tf.nn.relu)
		layer6 = tf.layers.dense(layer5,self.config.n_classes)
		return layer6
	def loss_op(self,pred):
		loss = tf.losses.mean_squared_error(labels=self.labels_placeholder,predictions=pred)
		#loss = loss
		return loss
	def training_op(self,loss):
		train_op = tf.train.AdamOptimizer().minimize(loss)
		return train_op

	def train_on_batch(self,sess,input_batch,labels_batch):
		feed= self.create_feed_dict(input_batch,labels_batch,self.config.dropout,True)
		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
		#train_writer.add_summary(rs, i)
		return loss
	def build(self):
		self.add_placerholders()
		self.pred=self.prediction_op()
		self.loss=self.loss_op(self.pred)
		self.train_op=self.training_op(self.loss)
	def fit(self,sess,train_x,train_y):
		self.train_on_batch(sess,train_x,train_y)
	def get_pred(self,sess,input_batch):
		feed = self.create_feed_dict(input_batch,None,1,0)
		pred=sess.run(self.pred,feed_dict=feed)
		return pred


def load_data(filename):
	tmp=np.load(filename)
	train_X=tmp['X']
	train_y=tmp['Y']
	return train_X,train_y

def dagger(sess,model):
	policy_name ='experts/Hopper-v1'
	policy_fn = load_policy.load_policy(policy_name+'.pkl')
	#print(policy_fn)
	env = gym.make(Config.envname)
	rollouts = 20
	observations = []
	actions = []
	for _ in range(rollouts):
		obs = env.reset()
		#print(obs.shape)
		done = False
		steps = 0
		while not done:
			action = model.get_pred(sess, obs[None, :])
			action_new = policy_fn(obs[None, :])
			obs, r, done, _ = env.step(action)
			#print(obs.shape)
			observations.append(obs)
			actions.append(action_new)
			steps += 1
			if steps >= Config.max_steps:
				break
	return np.array(observations),np.array(actions)

def main():
	file='10result.npz'
	trainx,trainy=load_data(file)
	config=Config()
	trainy=trainy.reshape(-1,config.n_classes)
	model = Model(config)
	init=tf.global_variables_initializer()
	shuffle_batch_x, shuffle_batch_y = tf.train.shuffle_batch(
	[trainx, trainy], batch_size=config.batch_size, capacity=10000,min_after_dequeue=5000, enqueue_many=True)
	with tf.Session() as sess:
		train_log_path='log'
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess, coord)
		losses=[]
		means=[]
		stds=[]
		sess.run(init)
		
		for i in tqdm.tqdm(range(config.itera)):
			try:
				for j in range(7820):#
					batchx,batchy=sess.run([shuffle_batch_x,shuffle_batch_y])
					loss =model.train_on_batch(sess,batchx,batchy)
					

					if j % 1000 == 0:
						new_x,new_y = dagger(sess,model)
						new_y = new_y.reshape(-1,Config.n_classes)
						trainx= tf.concat([trainx,new_x],axis=0)
						trainy= tf.concat([trainy,new_y],axis=0)
						print("step:", j, "loss:", loss)
					#	saver.save(sess,  "model/model_ckpt")	
					if j%100==0:
						losses.append([j,loss])

			except tf.errors.OutOfRangeError:
					print("")
			finally:
				coord.request_stop()
			coord.join(threads)

			policy_name ='experts/Hopper-v1'
			#envname ='Ant-v1'
			policy_fn = load_policy.load_policy(policy_name+'.pkl')
			env = gym.make(config.envname)
			rollouts = 20
			returns = []
			observations = []
			actions = []
			for _ in range(rollouts):
				obs = env.reset()
				done = False
				totalr = 0.
				steps = 0
				while not done:
					action = model.get_pred(sess, obs[None, :])
					obs, r, done, _ = env.step(action)
					totalr += r
					steps += 1
                        # if args.render:
					#env.render()
					if steps >= config.max_steps:
						break
				returns.append(totalr)
			means.append(np.mean(returns))
			stds.append(np.std(returns))
			print('mean of returns',np.mean(returns))
			print('std of returns ',np.std(returns))


		df = pd.DataFrame(losses,columns=['j','loss'])
		#print(df)
		sns.tsplot(data=df.loss)
		figname = config.envname+'dagger'
		plt.savefig(figname)
		#print(losses)
		#plt.show()
		df1 = pd.DataFrame([means,stds],index=['mean','std'])
		df1 = df1.T
		csvname = figname+str(config.train_itera)
		df.to_csv('loss_dagger.csv')
		df1.to_csv(csvname)


if __name__ == '__main__':
	main()
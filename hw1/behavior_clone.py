#*-*coding:utf-8*-*
#!/usr/bin/env python

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
#	def run_epoch(self,sess,):
#		n_minibatches = 1+len(train_example)/self.config.batch_size
#		for i,(train_x,train_y) in enumerate(M)

def load_data(filename):
	tmp=np.load(filename)
	train_X=tmp['X']
	train_y=tmp['Y']
	return train_X,train_y



#def build_mode(X,update):
#	self.X = tf.placeholder()
	#input_layer = tf.reshape(X,[-1,47,8,1])
	#conv1 = tf_util.conv2d(X,8,'conv1')
	#norm1 = tf_util.batchnorm(conv1,'norm1',update)
	#conv2 = tf_util.conv2d(norm1,16,'conv2')
	#norm2 = tf_util.batchnorm(conv2,'norm2',update)
	#conv3 = tfutil.conv2d(norm2,32,'conv3')
	#norm3 = tf_util.batchnorm(conv3,'norm3',update)
	#flat = tf_util.flattenallbut0(norm3)
#	dense1 = tf_util.dense(X,128,'dense1')
	#norm4 = tf_util.batchnorm(dense1,'norm4',update)
#	dropout1 = tf_util.dropout(dense1,0.5)
#	dense2 = tf_util.dense(dropout1,classes,'dense2')
#	return dense2
#X =tf.placeholder(tf.float32,shape=[None,feature])
#y = tf.placeholder(tf.float32, shape=[None,classes])
#def run_model(session,pred,loss,tran_op,Xd,yd,epoch=1,batch_size=64,print_every=100,training=None,plot_loss= False):
#	train_indice = np.arange(Xd.shape[0])
#	np.random.shuffle(train_indice)
#	training_now = training is not None

#	variables = [loss,tran_op]
	#if training:
	#		variables
#	iteration =0
#	for e in range(epoch):
#		losses =[]
		
#	return total_loss



def main():
	file='10result.npz'
	trainx,trainy=load_data(file)
	
	config=Config()
	#print(config.batch_size)
	trainy=trainy.reshape(-1,config.n_classes)
	model = Model(config)
	init=tf.global_variables_initializer()
	#saver= tf.saver()
	shuffle_batch_x, shuffle_batch_y = tf.train.shuffle_batch(
	[trainx, trainy], batch_size=config.batch_size, capacity=10000,min_after_dequeue=5000, enqueue_many=True)
	saver = tf.train.Saver()
	#print(config.batch_size)
	#print(shuffle_batch_x)
	#pred = build_mode(X,True)
	#loss = tf.nn.l2_loss(y-pred)
	#optimizer= tf.train.AdamOptimizer()
	#train = run_expert.main()
	#file ='10result.npz'
	#train = np.load(file)
	#print(train)
	#X_train=train['X']
	#y_train =train['Y']
	#y_train = y_train.reshape(-1,classes)
	#tran_op= optimizer.minimize(loss)
	with tf.Session() as sess:
		train_log_path='log'
		#merged = tf.summary.merge_all()
		#train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess, coord)
		losses=[]
		means=[]
		stds=[]
		reward=[]
		sess.run(init)
		import tqdm
		for i in tqdm.tqdm(range(config.itera)):
			#print(i)
			j=0
			try:
				for j in range(int(math.ceil(config.train_itera*trainx.shape[0]/config.batch_size))):

				#print(j)
					batchx,batchy=sess.run([shuffle_batch_x,shuffle_batch_y])
				#print(j)

			#idx = train_indice[start_idx:start_idx+batch_size]
					loss =model.train_on_batch(sess,batchx,batchy)
				#j += 1
					if j % 1000 == 0:
						print("step:", j, "loss:", loss)
						saver.save(sess,  "model/model_ckpt")	
			#action_batch_size = yd[idx].shape[0]

			#loss,_ = session.run(variables,feed_dict=feed_dict)

				#losses.append(loss)
					if j%100==0:
						losses.append([j,loss])
					#	print("iteration{0}: with minibatch training loss ={1:.3g} ".format(i/100,loss))

			except tf.errors.OutOfRangeError:
					print("")
			finally:
				coord.request_stop()
			coord.join(threads)
				#iteration +=1
				#total_loss = np.sum(losses)/Xd.shape[0]
				#print("epoch, overall loss ={0:3g}".format(total_loss))
				#if plot_loss:
				#	plt.plot(losses)
			#		plt.grid(True)
		#			plt.title("EPOCH {}losses".format(e+1))
	#				plt.xlabel("minibatch number ")
		#			plt.ylabel("minibatch loss")
		#print("training")
		#run_model(sess,pred,loss,tran_op,X_train,y_train,1,64,100,True,plot_loss=True)
		#print('validation')
		#run_model(sess,pred,loss,tran_op,X_val,y_val,1,64)
			env = gym.make(config.envname)
			rollouts = 20
			returns = []
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
		figname = config.envname
		plt.savefig(figname)
		#print(losses)
		#plt.show()
		df1 = pd.DataFrame([means,stds],index=['mean','std'])
		df1 = df1.T
		csvname = figname+str(config.train_itera)
		df.to_csv('loss.csv')
		df1.to_csv(csvname+'.csv')


if __name__ == '__main__':
	main()
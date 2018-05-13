#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import behavior_clone
import time
import pandas as pd
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str,default='experts/Humanoid-v1')
    parser.add_argument('envname', type=str,default='Humanoid-v1')
    parser.add_argument('--render', action='store_true',default='--render')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session() as sess:
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            #print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                #print(len(action[0]))
                #print(len(observations[0]))
                obs, r, done, _ = env.step(action) 
                #print(obs.shape)
                #print(done)
                totalr += r
                steps += 1
                #f args.render:
                #    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        i=0
        path= '/home/xinzhao/Dropbox/cs294/homework/hw1/'
        time1 = time.time()
        filename = path+str(args.num_rollouts)+args.envname
           
        np.savez(filename, X=expert_data['observations'],Y=expert_data['actions'])
        print('finish')

        #feature= 376
        #classes = 17
        print(expert_data['observations'].shape)
        print(expert_data['actions'].shape)
        df =pd.DataFrame([np.mean(returns),np.std(returns)],index = ['mean','std'])
        df = df.T
        df.to_csv('run_expert.csv')
        #X =tf.placeholder(tf.float32,shape=[None,feature])
        #y = tf.placeholder(tf.float32, shape=[None,classes])
        #X_train=expert_data['observations']
        #y_train =expert_data['actions']
        #y_train =y_train.reshape(-1,classes)
        #pred = behavior_clone.build_mode(X,True)
        #loss = tf_util.categorical_sample_logits(y_train-pred)
        #optimizer= tf.train.AdamOptimizer()
        #tran_op= optimizer.minimize(loss)
        #sess.run(tf.global_variables_initializer())
        #print("training")
        #behavior_clone.run_model(sess,pred,loss,tran_op,X_train,y_train,1,64,100,True)
        #print('validation')
       # behavior_clone.run_model(sess,pred,loss,tran_op,X_val,y_val,1,64)

if __name__ == '__main__':
    main()

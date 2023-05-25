import gym
import metaworld
import random
import ctypes
import mujoco_py
from metaworld.policies import SawyerHammerV2Policy
from metaworld.policies import SawyerAssemblyV2Policy


# libglew = ctypes.CDLL("libGLEW.so")
# libglew.glewInit()
#check if glewInit() is working

print(metaworld.ML1.ENV_NAMES)

ml1 = metaworld.ML1('hammer-v2')
env = ml1.train_classes['hammer-v2']()
# ml1 = metaworld.ML1('assembly-v2')
# env = ml1.train_classes['assembly-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)
env.reset()
#set simulation speed
env.model.opt.timestep = 0.05
#initialize obs
obs = env.reset()
policy = SawyerHammerV2Policy()
# policy = SawyerAssemblyV2Policy()

for i in range(500):
    env.render()
    #decide which action to take
    #let the agent to follow the sample action
    action = policy.get_action(obs)
    obs, reward, done, info = env.step(action) # take a random action
    #get the rgb image 
    # img = env.render()
    # print (obs.shape)
    if done:
        env.reset()
    print (obs.shape)
    # convert observation to image and display
    

# env.close()







#collect 6 demis 
#train image encoders, use train.yaml
 #bd_encoders.ymal
 #vision.py
 #train.py
 

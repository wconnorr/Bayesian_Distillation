import gym

def _make_cartpole():
  #def curry():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
  #return curry

def make_cartpole_vector_env(num_envs):
  return gym.vector.SyncVectorEnv([
      _make_cartpole
      for i in range(num_envs)])

def make_atari_vector_env(num_envs, envname):
  global atariname
  atariname = envname
  return SyncVectorEnv([
      _make_atari#(envname, simplify)
      for i in range(num_envs)])

# TODO: Try w/ ClipRewardEnv
def _make_atari():
  #def curry():
    env = gym.wrappers.AtariPreprocessing(gym.make(atariname), scale_obs=True)
    env = gym.wrappers.RecordEpisodeStatistics(gym.wrappers.FrameStack(env, 4))
    return env
  #return curry


def make_atari(atariname):
  #def curry():
    env = gym.wrappers.AtariPreprocessing(gym.make(atariname), scale_obs=True)
    env = gym.wrappers.RecordEpisodeStatistics(gym.wrappers.FrameStack(env, 4))
    return env
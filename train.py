import numpy as np
import time
import torch
import os
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from run.run_evaluate import run_test2
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

device = "cuda:0"

def run_dt():
    train_model()


def train_model():
    time_str = time.strftime("%Y-%m-%d_%H:%M:%d", time.localtime())
    writer = SummaryWriter("./log/"+time_str)
    state_dim = 16
    act_dim = 1
    if os.path.exists("./saved_model/DTtest/normalize_dict.pt"):
        replay_buffer = torch.load("./saved_model/DTtest/normalize_dict.pt")
    else:
        replay_buffer = EpisodeReplayBuffer(16, 1, "./data/trajectory/trajectory_data.csv")
        torch.save(replay_buffer,"./saved_model/DTtest/normalize_dict.pt")
        save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                            "saved_model/DTtest")
    # logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std)
    batch_size = 128
    size = len(replay_buffer.trajectories) // batch_size
    logger.info(f"Size: {size}")

    step_num = size * 5

    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size,num_workers=8)

    model.train()
    model.to(device=device)
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask in tqdm(dataloader):
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        rtg = rtg.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)
            
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        i += 1
        writer.add_scalar("Action_loss",train_loss,i)
            # logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
        model.scheduler.step()

        if i % size == 0:
            model.save_net("saved_model/DTtest/"+time_str+"/"+str(i//size)+"/")
            model.init_eval()
            # logger.info(f"Test action: {model.take_actions(test_state)}")
            with torch.no_grad():
                logger.info(f"Run Test")
                run_test2(writer,i//size,model)

def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()

import numpy as np
import math
import logging
import time
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.strategy.dt_bidding_strategy import DtBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
from torch.utils.tensorboard.writer import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

device = "cuda:0"

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

def run_test():
    """
    offline evaluation
    """

    data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
    env = OfflineEnv()
    agent = DtBiddingStrategy()

    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
    rewards = np.zeros(num_timeStepIndex)
    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }

    for timeStep_index in range(num_timeStepIndex):
        # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

        pValue = pValues[timeStep_index]
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:

            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])

        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)
        # logger.info(f'Timestep Index: {timeStep_index + 1} End')
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)

    logger.info(f'Total Reward: {all_reward}')
    logger.info(f'Total Cost: {all_cost}')
    logger.info(f'CPA-real: {cpa_real}')
    logger.info(f'CPA-constraint: {cpa_constraint}')
    logger.info(f'Score: {score}')

class offline_testor():
    def __init__(self,times,writer) -> None:
        self.times = times
        self.data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
        self.writer = writer

        self.keys, test_dict = self.data_loader.keys, self.data_loader.test_dict
        self.key = self.keys[0]
        self.num_timeStepIndex, self.pValues, self.pValueSigmas, self.leastWinningCosts = self.data_loader.mock_data(self.key)
        self.agent = PlayerBiddingStrategy()
        self.env = OfflineEnv()

    def test_step(self):
        self.agent.reset()

        rewards = np.zeros(self.num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

        for timeStep_index in range(self.num_timeStepIndex):
        # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

            pValue = self.pValues[timeStep_index]
            pValueSigma = self.pValueSigmas[timeStep_index]
            leastWinningCost = self.leastWinningCosts[timeStep_index]

            if self.agent.remaining_budget < self.env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:

                bid = self.agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])

            tick_value, tick_cost, tick_status, tick_conversion = self.env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)

            # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
            over_cost_ratio = max((np.sum(tick_cost) - self.agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = self.env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - self.agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            self.agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)
            # logger.info(f'Timestep Index: {timeStep_index + 1} End')

        all_reward = np.sum(rewards)
        all_cost = self.agent.budget - self.agent.remaining_budget
        cpa_real = all_cost / (all_reward + 1e-10)
        cpa_constraint = self.agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)

        return all_reward, all_cost, cpa_real, cpa_constraint, score
    
    def test(self,model,epoch):
        self.agent.model = model.to(device)
        all_reward = 0
        all_cost = 0
        cpa_real = 0 
        cpa_constraint = 0
        score = 0
        for _ in range(self.times):
            t_all_reward, t_all_cost, t_cpa_real, t_cpa_constraint, t_score = self.test_step()
            all_reward += t_all_reward / self.times
            all_cost += t_all_cost / self.times
            cpa_real += t_cpa_real / self.times
            cpa_constraint += t_cpa_constraint / self.times
            score += t_score / self.times
        
        print(score)
        self.writer.add_scalar("Test/Total Reward",all_reward,epoch)
        self.writer.add_scalar("Test/Total Cost",all_cost,epoch)
        self.writer.add_scalar("Test/CPA-real",cpa_real,epoch)
        self.writer.add_scalar("Test/CPA-constraint",cpa_constraint,epoch)
        self.writer.add_scalar("Test/Score",score,epoch)

if __name__ == '__main__':
    pass
    # time_str = time.strftime("%Y-%m-%d_%H:%M:%d", time.localtime())
    # writer = SummaryWriter("./log/"+time_str)
    # a = offline_testor(30,writer)
    # a.test(1,1)
    

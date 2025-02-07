import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))


import numpy as np
from mesa.examples.basic.boid_flockers.agents import Boid
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.space import ContinuousSpace
from mesa import Agent, Model
from sklearn.cluster import DBSCAN

from mesa.visualization import Slider, SolaraViz, make_space_component

# ---- Agent Definitions ---- #

class OriginalPostAgent(Agent):
    def __init__(self, model, keywords):
        super().__init__(model)
        self.keywords = keywords
        self.color = "red"  # Visualization: Red for original posts

class AdBotAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        # 确保初始化位置有效性
        self.pos = (np.random.uniform(0, model.space.x_max), 
                   np.random.uniform(0, model.space.y_max))
        self.direction = np.random.uniform(-0.1, 0.1, 2)  # 缩小初始方向范围
        self.speed = 1.5
        self.risk_factor = 0.5
        self.w_sep, self.w_ali, self.w_coh = 1.5, 0.8, 1.2
        self.color = "blue"
        self.target_post = None  # To track the targeted Original Post

    def detect_and_target(self):
        # Detect nearby Original Posts
        posts = [agent for agent in self.model.space.get_neighbors(self.pos, 20, False) if isinstance(agent, OriginalPostAgent)]
        if posts:
            self.target_post = posts[0]  # Target the first detected Original Post

    def move_toward_target(self):
        if self.target_post:
            direction_to_post = np.array(self.target_post.pos) - np.array(self.pos)
            norm = np.linalg.norm(direction_to_post)
            if norm != 0:
                self.direction = direction_to_post / norm  # Normalize
                self.pos = tuple(np.array(self.pos) + self.direction * self.speed)

    def step(self):
        if not self.target_post:
            self.detect_and_target()  # Detect nearby Original Posts
        else:
            self.move_toward_target()  # Move towards the targeted post
    

class ShillBotAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.engagement_rate = np.random.rand()
        self.color = "green"  # Visualization: Green for shill bots

    def step(self):
        # Engage with posts near AdBots
        neighbors = self.model.space.get_neighbors(self.pos, 5, False)
        ad_bots = [agent for agent in neighbors if isinstance(agent, AdBotAgent)]
        if ad_bots:
            self.engagement_rate += 0.1 * len(ad_bots)
            # Move closer to engage more
            direction_to_ads = np.mean([np.array(ad.pos) - np.array(self.pos) for ad in ad_bots], axis=0)
            norm = np.linalg.norm(direction_to_ads)
            if norm != 0:
                self.pos = tuple(np.array(self.pos) + (direction_to_ads / norm))
class UserAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.trust_score = 0.2
        self.skepticism_level = np.random.rand()
        self.emotion_intensity = 0
        self.color = "xkcd:amber"  # Visualization: Yellow for users

    def step(self):
        # Randomly browse posts
        self.pos = tuple(np.array(self.pos) + np.random.uniform(-1, 1, 2))
        
        neighbors = self.model.space.get_neighbors(self.pos, 5, False)
        ad_density = sum(1 for agent in neighbors if isinstance(agent, AdBotAgent))
        self.emotion_intensity += 0.02 * ad_density * self.trust_score  # Emotional shift

        social_proof = len(neighbors) / (1 + self.skepticism_level)
        self.trust_score = 1 - np.exp(-0.1 * social_proof)

class PlatformAI(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.detection_threshold = 0.7

    def step(self):
        positions = [
            agent.pos for agent in self.model.agents 
            if hasattr(agent, 'pos') and isinstance(agent.pos, (tuple, list)) and len(agent.pos) == 2
        ]

        positions = [pos for pos in positions if not any(np.isnan(pos))]

        if positions:
            positions = np.array(positions)
            labels = DBSCAN(eps=3, min_samples=5).fit_predict(positions)

            anomaly_score = sum(labels == -1) / len(labels)
            if anomaly_score > self.detection_threshold:
                self.model.detection_intensity += 0.1

class SocialFlockingModel(Model):
    def __init__(self, width=100, height=100, num_ads=20, num_shills=15, num_users=50,seed=None):
        super().__init__(seed=seed)
        self.space = ContinuousSpace(width, height, torus=False) 
        self.detection_intensity = 0.5
        self.consistency_threshold = 0.5
        self.topic_heatmap = np.random.rand(width, height)

        # Create Agents
        # Create Original Posts
        for i in range(5):
            agent = OriginalPostAgent(self, keywords=["promo", "discount", "deal"])
            self.space.place_agent(agent, (np.random.uniform(0, width), np.random.uniform(0, height)))


        # Create Ad Bots
        for i in range(num_ads):
            agent = AdBotAgent(self)
            self.space.place_agent(agent, (np.random.randint(0, width), np.random.randint(0, height)))

        # Create Shill Bots
        for i in range(num_shills):
            agent = ShillBotAgent(self)
            self.space.place_agent(agent, (np.random.randint(0, width), np.random.randint(0, height)))


        # Create Users
        for i in range(num_users):
            agent = UserAgent(self)
            self.space.place_agent(agent, (np.random.randint(0, width), np.random.randint(0, height)))

        # Platform AI
        self.platform_ai = PlatformAI(self)

    def step(self):
        self.agents.do("step")

def agent_portrayal(agent):
    if isinstance(agent, AdBotAgent):
        return {"marker": "v","color": agent.color, "size": 20, "alpha": 1}  
    elif isinstance(agent, ShillBotAgent):
        return {"marker": "^" ,"color": agent.color, "size": 15, "alpha": 1} 
    elif isinstance(agent, UserAgent):
        transparalphaency = max(0, min(1, 1 - agent.trust_score))
        return {"marker":"D","color": agent.color, "size": 10, "alpha": transparalphaency} 
    elif isinstance(agent, OriginalPostAgent):
        return {"color": agent.color, "size": 50, "alpha": 1}


# Model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "num_ads": Slider(
        label="Number of Ad Bots",
        value=50,
        min=10,
        max=200,
        step=10,
    ),
    "num_shills": Slider(
        label="Number of Shill Bots",
        value=20,
        min=5,
        max=100,
        step=5,
    ),
    "num_users": Slider(
        label="Number of Users",
        value=100,
        min=50,
        max=500,
        step=10,
    ),
    "width": Slider(
        label="Grid Width",
        value=100,
        min=50,
        max=200,
        step=10,
    ),
    "height": Slider(
        label="Grid Height",
        value=100,
        min=50,
        max=200,
        step=10,
    ),
}

# Create the model instance
model = SocialFlockingModel()

# Visualization setup
page = SolaraViz(
    model,
    components=[make_space_component(agent_portrayal=agent_portrayal, backend='matplotlib')],
    model_params=model_params,
    name="Social Flocking Simulation",
)

page  # noqa

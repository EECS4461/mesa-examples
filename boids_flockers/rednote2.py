import numpy as np
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.visualization import Slider, SolaraViz, make_space_component
from sklearn.cluster import DBSCAN
from textblob import TextBlob

# 二维空间维度配置 - 移除 time_window
SPACE_DIMENSIONS = {
    'topic_heat': (0, 200),    # X轴：话题热度
    'sentiment': (0, 200),     # Y轴：情感倾向
}

class OriginalPostAgent(Agent):
    def __init__(self, model):
        super().__init__(model) # 修正 Agent 初始化，添加 unique_id
        self.keywords = self.generate_keywords()
        self.base_heat = 1.0
        self.position = (
            np.random.uniform(*SPACE_DIMENSIONS['topic_heat']),
            np.random.uniform(40, 60),  # 初始情感中性
        )

    def generate_keywords(self):
        topics = ["fashion", "tech", "beauty", "lifestyle"]
        post = f"New trend in {np.random.choice(topics)}! " + \
               f"Limited {np.random.choice(['discount','deal','offer'])} available!"
        return TextBlob(post).noun_phrases

class AdBotAgent(Agent):
    def __init__(self, model):
        super().__init__(model) # 修正 Agent 初始化，添加 unique_id
        self.position = self.random_position()
        self.speed = 1.5
        self.cluster_size = 1
        self.target_post = None
        self.ad_content = ""

    def random_position(self):
        return (
            np.random.uniform(*SPACE_DIMENSIONS['topic_heat']),
            np.random.uniform(*SPACE_DIMENSIONS['sentiment']),
        )

    def find_target(self):
        neighbors = self.model.space.get_neighbors(self.position, 10) # Removed agent_type argument
        posts = [agent for agent in neighbors if isinstance(agent, OriginalPostAgent)] # Filter manually
        if posts:
            self.target_post = max(posts, key=lambda p: len(p.keywords))
            self.generate_ad()
            return True
        return False

    def generate_ad(self):
        keywords = self.target_post.keywords
        template = [
            "Check our {} collection!",
            "Best {} deals here!",
            "Limited {} offer!"
        ]
        self.ad_content = np.random.choice(template).format(
            ', '.join(keywords[:2]))

    def move(self):
        if self.target_post:
            target_vec = np.array(self.target_post.position) - np.array(self.position)
            if np.linalg.norm(target_vec) > 0:
                direction = target_vec / np.linalg.norm(target_vec)
                self.position = tuple(np.array(self.position) + direction * self.speed)

    def step(self):
        if not self.target_post and not self.find_target():
            self.position = self.random_position()
        else:
            self.move()
        self.cluster_size = 1 + len(self.model.space.get_neighbors(self.position, 5))

class ShillBotAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.position = self.random_position() # Now this will work!
        self.velocity = np.random.rand(2) - 0.5
        self.weights = np.array([1.5, 0.8, 1.2])  # sep, ali, coh

    def random_position(self):
        return (
            np.random.uniform(*SPACE_DIMENSIONS['topic_heat']),
            np.random.uniform(*SPACE_DIMENSIONS['sentiment']),
        )

    def update_weights(self):
        detection = self.model.detection_intensity
        self.weights = np.array([
            1.5 * (1 + detection),
            0.8 / (0.1 + detection),
            1.2 * (1 - detection)
        ])

    def flocking_behavior(self, neighbors):
        shill_neighbors = [n for n in neighbors if isinstance(n, ShillBotAgent)] # Filter for ShillBotAgent neighbors
        positions = np.array([n.position for n in shill_neighbors]) # Use only ShillBotAgent positions

        if not shill_neighbors: # Handle case where there are no ShillBotAgent neighbors
            return np.array([0, 0]) # Or return a zero vector, indicating no flocking influence


        # 分离向量
        sep_vec = -np.mean(positions - self.position, axis=0)

        # 对齐向量
        ali_vec = np.mean([n.velocity for n in shill_neighbors], axis=0) # Use only ShillBotAgent velocities

        # 聚合向量
        coh_vec = np.mean(positions, axis=0) - self.position

        return self.weights[0]*sep_vec + self.weights[1]*ali_vec + self.weights[2]*coh_vec

    def step(self):
        self.update_weights()
        neighbors = self.model.space.get_neighbors(self.position, 10)
        if neighbors:
            flock_vec = self.flocking_behavior(neighbors)
            self.velocity = 0.2*self.velocity + 0.8*flock_vec
            self.position = tuple(np.array(self.position) + self.velocity)
        self.position = tuple(np.clip(self.position,
                                        [d[0] for d in SPACE_DIMENSIONS.values()],
                                        [d[1] for d in SPACE_DIMENSIONS.values()]))

class UserAgent(Agent):
    def __init__(self, model):
        super().__init__(model) # 修正 Agent 初始化，添加 unique_id
        self.position = self.random_position()
        self.trust = 0.5
        self.emotion = 0.5

    def random_position(self):
        return (
            np.random.uniform(*SPACE_DIMENSIONS['topic_heat']),
            np.random.uniform(*SPACE_DIMENSIONS['sentiment']),
        )

    def update_emotion(self):
        neighbors_ads = self.model.space.get_neighbors(self.position, 5)
        ads = [agent for agent in neighbors_ads if isinstance(agent, AdBotAgent)]
        ad_influence = len(ads) * 0.02  # **Correct definition of ad_influence**
        social_proof = np.mean([u.emotion for u in
                                   self.model.space.get_neighbors(self.position, 3) # Removed agent_type argument here as well, consistent with previous fixes
                                   if isinstance(u, UserAgent)] # Manual filtering for UserAgent neighbors
                                or [0.5])
        self.emotion = np.clip(
            self.emotion + ad_influence*self.trust + 0.1*(social_proof - 0.5),
            0, 1
        )
        self.trust *= 0.99 if ad_influence > 2 else 1.01

    def step(self):
        self.position = tuple(np.array(self.position) +
                                    np.random.uniform(-0.5, 0.5, 2)) # 修正为 2D 随机移动
        self.update_emotion()

class PlatformAI:
    def __init__(self, model):
        self.model = model
        self.boost_threshold = 0.6
        self.limit_threshold = 0.8

    def analyze_engagement(self):
        all_ads = [a for a in self.model.agents if isinstance(a, AdBotAgent)]
        positions = np.array([a.position for a in all_ads])

        if len(positions) > 10:
            # 第一阶段：推流算法
            clustering = DBSCAN(eps=5, min_samples=3).fit(positions) # 修正为 2D 聚类
            cluster_ratio = len(np.unique(clustering.labels_)) / len(positions)

            if cluster_ratio < self.boost_threshold:
                self.boost_posts(positions)

            # 第二阶段：限流检测
            if cluster_ratio > self.limit_threshold:
                self.limit_flow(clustering.labels_)

    def boost_posts(self, positions):
        heat_center = np.mean(positions[:, 0])
        self.model.heat_modifier = 1.5 + 0.5 * np.sin(self.model.steps/10)

    def limit_flow(self, labels):
        for agent, label in zip(self.model.agents, labels):
            if isinstance(agent, AdBotAgent) and label != -1:
                agent.speed *= 0.7
                agent.cluster_size = max(1, agent.cluster_size-2)

class SocialMediaModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.space = ContinuousSpace(
            SPACE_DIMENSIONS['topic_heat'][1],  # x_max (话题热度的最大值)
            SPACE_DIMENSIONS['sentiment'][1], # y_max (情感倾向的最大值)
            torus=False
        )
        self.platform_ai = PlatformAI(self)
        self.heat_modifier = 1.0
        self.detection_intensity = 0.5

        # 初始化代理
        self.create_agents(OriginalPostAgent, kwargs.get('num_op', 10))
        self.create_agents(AdBotAgent, kwargs.get('num_ads', 20))
        self.create_agents(ShillBotAgent, kwargs.get('num_shills', 40))
        self.create_agents(UserAgent, kwargs.get('num_users', 50))

    def create_agents(self, agent_type, num):
        for _ in range(num):
            agent = agent_type(self)
            self.space.place_agent(agent, agent.position)

    def step(self):
        self.agents.do("step")
        self.platform_ai.analyze_engagement()
        self.detection_intensity = min(1.0, self.detection_intensity + 0.01)

def agent_portrayal(agent):
    base = {
        "position": agent.position, # 修正为 2D position
        "size": 10,
        "alpha": 1
    }

    if isinstance(agent, OriginalPostAgent):
        return {"marker": "*", "color": "#FF0000", "size": 40, "alpha": 1}

    elif isinstance(agent, AdBotAgent):
        return {
            "marker": ".",
            "color": "#1E90FF",
            "size": 15 + agent.cluster_size,
            "alpha": max(0.3, agent.cluster_size/10)
        }

    elif isinstance(agent, ShillBotAgent):
        neighbors = len(agent.model.space.get_neighbors(agent.position, 5))
        return {
            "marker": "^",
            "color": "#00FF00" if neighbors < 5 else "#8A2BE2",
            "size": 10 + neighbors,
            "alpha": 0.8
        }

    elif isinstance(agent, UserAgent):
        return {
            "marker": "D",
            "color": "#FFD700",
            "size": 10,
            "alpha": max(0.2, agent.emotion)
        }

    return base

# 可视化参数配置
model_params = {
    "num_op": Slider("Origin Post", 20, 10, 100),
    "num_ads": Slider("Ad Bots", 20, 10, 100),
    "num_shills": Slider("Shill Bots", 15, 5, 50),
    "num_users": Slider("Users", 50, 20, 200),
    "detection": Slider("Detection", 0.5, 0.1, 1.0, 0.1)
}

viz = SolaraViz(
    SocialMediaModel,
    components=[make_space_component(agent_portrayal)], # 移除空间维度参数，默认为模型空间维度
    model_params=model_params,
    name="Social Media Coordination",
    # space_dims=SPACE_DIMENSIONS # 移除 space_dims 参数，组件会自动从模型空间获取
)

model = SocialMediaModel()

page = SolaraViz(
    model,
    components=[make_space_component(agent_portrayal)], # 移除空间维度参数，默认为模型空间维度
    model_params=model_params,
    name="Social Media Coordination",
    # space_dims=SPACE_DIMENSIONS # 移除 space_dims 参数，组件会自动从模型空间获取
)

page  # noqa
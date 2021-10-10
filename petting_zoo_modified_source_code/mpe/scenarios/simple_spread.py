import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.target = world.landmarks[i]

        return world

    def reset_world(self, world, np_random):
        size = 0.5*np.sqrt(len(world.agents))
        
        if world.agents[0].color is None:
            # random properties for landmarks
            for i, agent in enumerate(world.agents):
                agent.color = np_random.uniform(-1, +1, 3)
                agent.target.color = agent.color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-size, +size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-size, +size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_on_target(self, agent):
        delta_pos = agent.state.p_pos - agent.target.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + agent.target.size
        return True if dist < dist_min else False
    
    def action_penalty(self, agent):
        if agent.action.u[0] != 0 or agent.action.u[1] != 0:
            return True
        else:
            return False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.target.state.p_pos)))
        if self.action_penalty(agent):
            rew -= 1
        if self.is_on_target(agent):
            rew += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def assess_world(self, world):
        n_collisions = 0
        n_zero_action = 0
        n_target_reached = 0

        for a1 in world.agents:
            if not self.action_penalty(a1):
                n_zero_action += 1
            if self.is_on_target(a1):
                n_target_reached += 1
            for a2 in world.agents:
                if a1 != a2:
                    if self.is_collision(a1, a2):
                        n_collisions += 1
        n_collisions /= 2
        n_agents = len(world.agents)
        completion_rate = n_target_reached / n_agents
        collision_rate = n_collisions / n_agents
        zero_action_rate = n_zero_action / n_agents
        return completion_rate, collision_rate, zero_action_rate

    def global_reward(self, world):
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        """ # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos) """
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]) # + entity_pos + other_pos + comm)

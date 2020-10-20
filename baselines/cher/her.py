import numpy as np
import random
import baselines.cher.config_curriculum as config_cur
import math
from sklearn.neighbors import NearestNeighbors
from gym.envs.robotics import rotations


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):

    # reward_fun is defined in configure_her of config.py
    print('calling init of make_sample_her_transitions')
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    print('future_p is', future_p)

    def curriculum(transitions, batch_size_in_transitions, batch_size):

        print('calling curriculum of her', 'batch_size_in_transitions, batch_size', batch_size_in_transitions, batch_size)
        # print(transitions['g'].shape)

        for key, value in transitions.items():
            print('key, value of curriculum in her', key, value.shape)

        #select a series states to explore
        sel_list = lazier_and_goals_sample_kg(transitions['g'], transitions['ag'], transitions['o'], batch_size_in_transitions)
        print('sel_list of curriculum in her is', sel_list.shape)


        transitions = {
            key: transitions[key][sel_list].copy()
            for key in transitions.keys()
        }
        config_cur.learning_step += 1
        return transitions

    def fa(k, a_set, v_set, sim, row, col):
        #k is a series of sub-goas, once a sub-goal ia achievd, the reward is updated to the maximal reward
        if len(a_set) == 0:
            # print('a_set of fa in her is none')
            init_a_set = []
            marginal_v = 0
            for i in v_set:
                max_ki = 0
                if k == col[i]:
                    max_ki = sim[i]
                init_a_set.append(max_ki)
                marginal_v += max_ki
                # print(i, 'marginal_v', marginal_v, 'init_a_set', init_a_set)
            return marginal_v, init_a_set

        # print('a_set of fa in her is not none')
        new_a_set = []
        marginal_v = 0
        for i in v_set:
            sim_ik = 0
            if k == col[i]:
                sim_ik = sim[i]

            if sim_ik > a_set[i]:
                max_ki = sim_ik
                new_a_set.append(max_ki)
                marginal_v += max_ki - a_set[i]
            else:
                new_a_set.append(a_set[i])
        return marginal_v, new_a_set

    def lazier_and_goals_sample_kg(goals, ac_goals, obs, batch_size_in_transitions):
        if config_cur.goal_type == "ROTATION":
            goals, ac_goals = goals[..., 3:], ac_goals[..., 3:]

        num_neighbor = 1
        #find each point's closet point by a kd-tree
        #kgraph get the closest state of each state
        kgraph = NearestNeighbors(n_neighbors=num_neighbor, algorithm='kd_tree',metric='euclidean').fit(goals).kneighbors_graph(
                mode='distance').tocoo(copy=False)

        # print('kgraph of lazier_and_goals_sample_kg in her is', kgraph)

        row = kgraph.row
        col = kgraph.col

        # print('row and col of her are', row, col)

        sim = np.exp( -np.divide(np.power(kgraph.data, 2), np.mean(kgraph.data)**2))
        # print('sim of her is', np.mean(kgraph.data)**2, sim)

        delta = np.mean(kgraph.data)
        # print('delta of her is', delta)

        sel_idx_set = []
        #index from 0 to learning_candidates size
        idx_set = [i for i in range(len(goals))]
        # print('idx_set of her', idx_set)

        balance = config_cur.fixed_lambda
        print('balance of her', balance)

        if int(balance) == -1:
            balance = math.pow( 1 + config_cur.learning_rate, config_cur.learning_step) * config_cur.lambda_starter
            print('balance of her is updated', balance, 'config_cur.learning_step', config_cur.learning_step)

        v_set = [i for i in range(len(goals))]
        max_set = []
        for i in range(0, batch_size_in_transitions):
            sub_size = 3
            # randomly get sub_size indexes ranging from 0 to learning_candidates size
            sub_set = random.sample(idx_set, sub_size)
            # print('i and sub_set of her are', sub_set)
            sel_idx = -1
            max_marginal = float("-inf")  #-1 may have an issue
            #explore sub_size times to select an index of the maximal reward
            for j in range(sub_size):
                k_idx = sub_set[j]
                #marginal_v, getting the value of the states via kd-tree, which is the importance of achiving some sub-goals
                marginal_v, new_a_set = fa(k_idx, max_set, v_set, sim, row, col)
                # print(j, 'marginal_v, new_a_set of her', marginal_v, len(new_a_set))
                #euclidean distance
                euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])
                # print(j, 'euc of her', euc)
                #marginal_v is becoming smaller once
                marginal_v = marginal_v - balance * euc
                if marginal_v > max_marginal:
                    sel_idx = k_idx
                    max_marginal = marginal_v
                    max_set = new_a_set

            idx_set.remove(sel_idx)
            sel_idx_set.append(sel_idx)
        return np.array(sel_idx_set)

    # does not use it: from gym https://github.com/openai/gym/blob/master/gym/envs/robotics/hand/manipulate.py#L87
    def _goal_rot_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7
        d_rot = np.zeros_like(goal_b[..., 0])
        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a,
                                       rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
        return d_rot

    # does not use it
    def lazier_and_goals_sample(goals, ac_goals, obs,
                                batch_size_in_transitions):
        init = []
        init.append(goals[0])
        sel_idx_set = set([0])
        idx_set = [i for i in range(len(goals))]
        idx_set.remove(0)
        balance = 1.0
        #balance = config_cur.learning_down + config_cur.learning_rate * config_cur.learning_step / config_cur.total_learning_step
        #balance = math.pow(1 + config_cur.learning_rate, config_cur.learning_step)*config_cur.lambda_starter
        balance = math.pow(1 + config_cur.learning_rate,
                           config_cur.learning_step)
        for i in range(1, batch_size_in_transitions):
            max_dist = np.NINF  #-100.
            sel_idx = -1
            sub_size = 3
            sub_set = random.sample(idx_set, sub_size)
            for j in range(sub_size):
                ob = obs[sub_set[j]]
                gripper_pos = ob[0:3]
                object_pos = ob[3:6]
                dist = get_distance(goals[sub_set[j]], init)
                comb_dist = dist / len(init) - balance * np.linalg.norm(
                    goals[sub_set[j]] - ac_goals[sub_set[j]]
                ) - balance * np.linalg.norm(gripper_pos - object_pos)
                #comb_dist = -balance * np.linalg.norm(goals[sub_set[j]]-ac_goals[sub_set[j]])
                if comb_dist > max_dist:
                    max_dist = comb_dist
                    sel_idx = sub_set[j]
            init.append(goals[sel_idx])
            idx_set.remove(sel_idx)
            sel_idx_set.add(sel_idx)
        return np.array(list(sel_idx_set))

    # does not use it
    def get_distance(p, init_set):
        dist = 0.
        for i in range(len(init_set)):
            dist += np.linalg.norm(p - init_set[i])
        return dist

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        print('calnilg _sample_her_transitions of her')

        T = episode_batch['u'].shape[1]
        # print('T of _sample_her_transitions is', T)

        rollout_batch_size = episode_batch['u'].shape[0]
        # print('rollout_batch_size of her', rollout_batch_size)

        batch_size = config_cur.learning_candidates
        # print('batch_size of her', batch_size)

        # Select which episodes and time steps to use, idxs:[batch_size,] ranging from 0 to rollout_batch_size
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # print('episode_idxs of her is', len(episode_idxs), episode_idxs)

        #get a batch of size [batch_size,], which ranging from 0 to T
        t_samples = np.random.randint(T, size=batch_size)
        # print('t_samples of her is',  t_samples.shape, t_samples)

        # for key in episode_batch.keys():
        #     print('key of her', key, episode_batch[key].shape, episode_batch[key][episode_idxs, t_samples].shape)

        #get a new transition of size [batch_size, dims], each timestep is randomly sampled from all the rollouts
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }
        # print('transitions of her', transitions['o'].shape)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        #np.random.uniform(size=batch_size): get an array index of size [batch_size,]
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # print('her_indexes of her', len(her_indexes[0]), her_indexes)

        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        # print('future_offset', future_offset)
        # print('T - t_samples', T, t_samples)

        future_offset = future_offset.astype(int)
        #get offset indexes
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # print('t_samples + 1 + future_offset is', (t_samples + 1 + future_offset).shape, t_samples + 1 + future_offset)
        # print('her_indexes is', her_indexes[0].shape, her_indexes)
        # print('future_t is', future_t.shape, future_t)



        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        #episode_idxs is rollout index, episode_idxs[her_indexes] choose some rollouts less than batch_size
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        # print('episode_idxs[her_indexes]', episode_idxs[her_indexes].shape)
        print('future_ag of her', future_ag.shape)
        transitions['g'][her_indexes] = future_ag

        #assert batch_size_in_transitions == 64
        if batch_size_in_transitions != config_cur.learning_selected:
            batch_size_in_transitions = config_cur.learning_selected

        # curriculum learning process
        transitions = curriculum(transitions, batch_size_in_transitions, batch_size)
        batch_size = batch_size_in_transitions

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            print('key and value of her', key, value.shape)
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        # print('reward_params of her', reward_params)

        reward_params['info'] = info
        print('keys in reward_params of her', reward_params.keys())
        # print('reward_params[\'info\']', reward_params['info'].keys(), reward_params['info']['is_success'].shape)

        # return values of 0 or -1 in size of [batcg_size,]
        transitions['r'] = reward_fun(**reward_params)
        # print('reward of transitions in her', transitions['r'])

        # for k in transitions.keys():
        #     print(k,'*transitions[k].shape[1:]', *transitions[k].shape[1:], transitions[k].shape[1:], transitions[k].shape)

        # for k in transitions.keys():
        #     print(k,'transitions[k].shape', transitions[k].shape)

        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }

        # for k in transitions.keys():
        #     print(k,'reshape transitions[k].shape', transitions[k].shape)

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions

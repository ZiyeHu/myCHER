import tensorflow as tf
from baselines.her.util import store_args, nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # print('self.o_tf', self.o_tf)

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        print('calling ActorCritic:', 'o.shape', o.shape, 'g.shape', g.shape)

        input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        print('input_pi.shape', input_pi.shape)

        # Networks.
        with tf.variable_scope('pi'):
            print('prepare pi','self.hidden', self.hidden, 'self.layers', self.layers, 'self.dimu', self.dimu, 'create a simple network', [self.hidden] * self.layers + [self.dimu])
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu]))
            print('self.max_u', self.max_u, 'self.pi_tf', self.pi_tf.shape)
        with tf.variable_scope('Q'):
            # for policy training
            print('prepare for policy training')
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            print('input_Q.shape', input_Q.shape , [self.hidden] * self.layers + [1])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            print('self.Q_pi_tf.shape', self.Q_pi_tf.shape)


            # for critic training
            print('prepare for critic training')
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
            print('self.Q_tf.shape', self.Q_tf.shape)

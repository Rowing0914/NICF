""" To avoid the unnecessary import I've transferred some functions from other files """
import copy as cp
import numpy as np
from env import env as Env

THRESHOLD = 300
tau = 0
memory = []


def str2bool(str=""):
    """ Taken from launch.py """
    str = str.lower()
    if str.__contains__("yes") or str.__contains__("true") or str.__contains__("y") or str.__contains__("t"):
        return True
    else:
        return False


def arg_parser():
    """ Taken from util.py """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """ Taken from launch.py """
    parser = arg_parser()
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-environment', type=str, default="env")
    parser.add_argument('-data_path', type=str, default="./data/data")
    parser.add_argument('-agent', type=str, default="training methods")
    parser.add_argument('-FA', type=str, default="function approximation")
    parser.add_argument('-T', dest='T', type=int, default=3, help="time_step")
    parser.add_argument('-ST', dest='ST', type=eval, default="[10,30,60,120]", help="evaluation_time_step")
    parser.add_argument('-gpu_no', dest='gpu_no', type=str, default="0", help='which gpu for usage')
    parser.add_argument('-latent_factor', dest='latent_factor', type=int, default=10, help="latent factor")
    parser.add_argument('-learning_rate', dest='learning_rate', type=float, default=0.01, help="learning rate")
    parser.add_argument('-training_epoch', dest='training_epoch', type=int, default=30000, help="training epoch")
    parser.add_argument('-rnn_layer', dest='rnn_layer', type=int, default=1, help="rnn_layer")
    parser.add_argument('-inner_epoch', dest='inner_epoch', type=int, default=50, help="rnn_layer")
    parser.add_argument('-batch', dest='batch', type=int, default=128, help="batch_size")
    parser.add_argument('-gamma', dest='gamma', type=float, default=0.0, help="gamma")
    parser.add_argument('-clip_param', dest='clip_param', type=float, default=0.2, help="clip_param")
    parser.add_argument('-restore_model', dest='restore_model', type=str2bool, default="False", help="")
    parser.add_argument('-num_blocks', dest='num_blocks', type=int, default=2, help="")
    parser.add_argument('-num_heads', dest='num_heads', type=int, default=1, help="")
    parser.add_argument('-dropout_rate', dest='dropout_rate', type=float, default=0.1, help="")
    return parser


def convert_item_seq2matrix(item_seq):
    """ Taken from agents/Train.py """
    max_length = max([len(item) for item in item_seq])
    matrix = np.zeros((max_length, len(item_seq)), dtype=np.int32)
    for x, xx in enumerate(item_seq):
        for y, yy in enumerate(xx):
            matrix[y, x] = yy
    target_index = list(zip([len(i) - 1 for i in item_seq], range(len(item_seq))))
    return matrix, target_index


if __name__ == '__main__':
    # === From launch.py ===
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    args.data_path = "/home/norio0925/Desktop/workspace/RL_RS/NICF/data/data"

    _type = "training"
    env = Env(args=args)

    args.user_num = env.user_num
    args.item_num = env.item_num
    args.utype_num = env.utype_num

    # === From agents/Train.py ===
    if _type == "training":
        selected_users = np.random.choice(env.training, (args.inner_epoch,))
    elif _type == "validation":
        selected_users = env.validation
    elif _type == "evaluation":
        selected_users = env.evaluation
    elif _type == "verified":
        selected_users = env.training
    else:
        selected_users = range(1, 3)
    infos = {item: [] for item in args.ST}
    used_actions = []
    for uuid in selected_users:
        actions = {}
        rwds = 0
        done = False
        state = env.reset_with_users(uid=1)
        while not done:
            data = {"uid": [state[0][1]]}
            for i in range(6):
                p_r, pnt = convert_item_seq2matrix([[0] + [item[0] for item in state[1] if item[3]["rate"] == i]])
                data["p" + str(i) + "_rec"] = p_r
                data["p" + str(i) + "t"] = pnt
            # policy = self.fa["model"].predict(self.fa["sess"], data)[0]
            if _type == "training":
                if np.random.random() < 5 * THRESHOLD / (THRESHOLD + tau):
                    policy = np.random.uniform(0, 1, (args.item_num,))
                for item in actions:
                    policy[item] = -np.inf
                action = np.argmax(policy[1:]) + 1
            else:
                for item in actions:
                    policy[item] = -np.inf
                action = np.argmax(policy[1:]) + 1
            s_pre = cp.deepcopy(state)
            state_next, rwd, done, info = env.step(action)
            if _type == "training":
                memory.append([s_pre, action, rwd, done, cp.deepcopy(state_next)])
            actions[action] = 1
            rwds += rwd
            state = state_next
            if len(state[1]) in args.ST:
                infos[len(state[1])].append(info)
        used_actions.extend(list(actions.keys()))

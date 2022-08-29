from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import time
import sys

import numpy as np
from utils import command_parser
from utils.misc_util import Logger

from utils.class_finder import model_class, agent_class, optimizer_class
from utils.model_util import ScalarMeanTracker
from utils.data_utils import loading_scene_list
# from main_eval import main_eval
# from full_eval import full_eval

from runners import collect_data

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    setproctitle.setproctitle("Train/Test Manager")
    args = command_parser.parse_arguments()

    # records related
    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    work_dir = os.path.join(args.work_dir, '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not args.no_logger:
        log_file = os.path.join(work_dir, 'train.txt')
        sys.stdout = Logger(log_file, sys.stdout)
        sys.stderr = Logger(log_file, sys.stderr)

    # start training preparation steps
    if args.remarks is not None:
        print(args.remarks)
    print('Training started from: {}'.format(
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    )

    args.learned_loss = False
    args.num_steps = 50

    args.data_dir = os.path.expanduser('/disk4/nuri/AI2Thor_offline_data_2.0.3/')
    scenes = loading_scene_list(args)
    print(args)

    init_agent = agent_class(args.agent_type)

    np.random.seed(args.seed)
    random.seed(args.seed)

    args.gpu_ids = [-1]

    train_total_ep = 0
    n_frames = 0
    collect_data(args, init_agent, scenes)

    # train_thin = args.train_thin
    # train_scalars = ScalarMeanTracker()

    start_time = time.time()

    lr = args.lr

    while train_total_ep < args.max_ep:

        train_result = train_res_queue.get()
        train_scalars.add_scalars(train_result)
        train_total_ep += 1
        n_frames += train_result['ep_length']

        if (train_total_ep % args.ep_save_freq) == 0:
            print('{}: {}: {}'.format(
                train_total_ep, n_frames, time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
            )
        if args.test_speed and train_total_ep % 10000 == 0:
            print('{} ep/s'.format(10000 / (time.time() - start_time)))
            start_time = time.time()


if __name__ == "__main__":
    main()

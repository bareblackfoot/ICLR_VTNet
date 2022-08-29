from __future__ import division

import time
import torch

from datasets.constants import AI2THOR_TARGET_CLASSES

import setproctitle

from datasets.data import num_to_name
from models.model_io import ModelOptions

from agents.random_agent import RandomNavigationAgent

import random

from runners.train_util import (
    compute_loss,
    new_episode,
    run_optimal_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)


def collect_data(
        args,
        initialize_agent,
        scenes,
):
    targets = AI2THOR_TARGET_CLASSES[args.num_category]

    # torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    # if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)
    rank = 0
    gpu_id = 0
    player = initialize_agent(None, args, rank, scenes, targets, gpu_id=gpu_id)
    compute_grad = not isinstance(player, RandomNavigationAgent)

    model_options = ModelOptions()

    episode_num = 0
    while episode_num < 10000:#args.num_episodes:
        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        player.episode.episode_times = episode_num
        new_episode(args, player)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            total_reward = run_optimal_episode(player, args, total_reward, model_options, True)
        results = {
            'done_count': player.episode.done_count,
            'ep_length': player.eps_len,
            'success': int(player.success),
            # 'seen_percentage': player.episode.seen_percentage,
            'tools': {
                'scene': player.episode.scene,
                'target': player.episode.task_data,
                'states': player.episode.states,
                'action_outputs': player.episode.action_outputs,
                'action_list': [int(item) for item in player.episode.actions_record],
                'detection_results': player.episode.detection_results,
                'success': player.success,
            }
        }
        json.dump(results, open(os.path.join(args.save_dir, '{}_{}.json'.format(player.episode.scene, player.episode.task_data)), 'w'))
        reset_player(player)
        episode_num = (episode_num + 1) % len(args.scene_types)

    player.exit()

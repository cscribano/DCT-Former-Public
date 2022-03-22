# -*- coding: utf-8 -*-
# ---------------------

import torch.distributed as dist
from torch.distributed import *
import sh

__all__ = ['AverageMeter', 'reduce_tensor']

def save_git_stuff(exp_dir):
    # Save the git hash
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_128:
        logger.info(
            "Seems like the code is not running from"
            " within a git repo, so hash will"
            " not be stored. However, it"
            " is strongly advised to use"
            " version control."
        )
    # And the git diff
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_129:
        logger.info(
            "Seems like the code is not running from"
            " within a git repo, so diff will"
            " not be stored. However, it"
            " is strongly advised to use"
            " version control."
        )

def get_time_diff_hours(now, start_marker):
    """return difference between 2 time markers in hours"""
    time_diff = now - start_marker
    time_diff_hours = time_diff / 3600
    return time_diff_hours


def is_time_to_exit(now, max_steps, cnf):
    time_diff_hours = get_time_diff_hours(now, cnf.experiment.exp_start_marker)

    ttt = cnf.experiment.get("total_training_time", None)
    tts = cnf.experiment.get("max_steps", None)

    # if passed max_pretrain_hours, then exit
    if ttt is not None:
        if time_diff_hours > ttt:
            return True

    # If exceeded max steps, then exit
    if tts is not None:
        if max_steps >= tts:
            return True

    return False

def is_time_to_finetune(now, start_marker, time_markers, total_time):
    if time_markers is None:
        return False
    time_diff_hours = get_time_diff_hours(now, start_marker)
    if len(time_markers) > 0 and time_diff_hours / total_time > time_markers[0]:
        time_markers.pop(0)
        return True
    else:
        return False


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

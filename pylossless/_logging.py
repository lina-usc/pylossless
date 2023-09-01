import time

from mne.utils import logger
from functools import wraps

CONF_MAP = {"run_staging_script": "staging_script", "find_breaks": "find_breaks"}


def _is_step_run(func, instance):
    """Check if step was run."""
    if func.__name__ in ["run_staging_script", "find_breaks"]:
        step = CONF_MAP[func.__name__]
        return (
            True if step not in instance.config or not instance.config[step] else False
        )
    else:
        return False


def lossless_logger(func=None, *, message=None, verbose=True):
    """Handle start and completion logging for pipeline steps.

    Parameters
    ----------
    func
        pipeline method to be logged
    message : str
        message about the step being run to provide to user
    verbose : bool
        if True, print logging message. if False, suppress logging
        message.
    """
    if func is None:
        return lambda f: lossless_logger(f, message=message)

    @wraps(func)
    def wrapper(*args, message=None, **kwargs):
        start_time = time.time()
        instance = args[0]
        skip = _is_step_run(func, instance)
        this_step = message if message is not None else func.__name__
        if skip:
            logger.info(f"LOSSLESS: Skipping {this_step}")
            return
        elif verbose:
            logger.info(f"LOSSLESS: 🚩 {this_step}.")
        result = func(*args, **kwargs)
        end_time = time.time()
        dur = f"{end_time - start_time:.2f}"
        if verbose:
            logger.info(f"LOSSLESS: 🏁 Finished {this_step} after {dur}" " seconds.")
        return result

    return wrapper


def lossless_time(func):
    """Log the time of a full pipeline run."""

    def wrapper(*args, **kwargs):
        logger.info(" ⏩ LOSSLESS: Starting Pylossless Pipeline.")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        dur = f"{(end_time - start_time) / 60:.2f}"
        logger.info(f"  ✅ LOSSLESS: Pipeline completed! took {dur} minutes.")
        return result

    return wrapper

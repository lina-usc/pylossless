"""Python port of EEG-IP-L pipeline for preprocessing EEG."""

from . import pipeline, bids, config, flagging, utils
from .pipeline import LosslessPipeline

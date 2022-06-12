import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict
from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from model import BertForMultiLabelClassification
from utils import (
    init_logger,
    set_seed,
    compute_metrics
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

logger = logging.getLogger(__name__)


def main(cli_args):
    # Read from config file and make args
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = GoEmotionsProcessor(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None
    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--taxonomy", type=str, required=True, help="Taxonomy (original, ekman, group)")

    cli_args = cli_parser.parse_args()

    main(cli_args)

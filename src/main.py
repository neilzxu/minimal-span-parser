from typing import Dict
import argparse
import itertools
import logging
import os
import os.path
import sys
import time

import dynet as dy
import numpy as np

import evaluate
import parse
import trees
import vocabulary
import yaml

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger('minimal-span-parser')
logger.setLevel(LOGLEVEL)
logger.addHandler(logging.StreamHandler(sys.stdout))


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def load_language_embeddings(lang_file) -> Dict[str, np.ndarray]:
    arr = np.load(lang_file)
    langs = arr['langs']
    features = arr['data']

    result = {
        langs[i][:2]: features[i].reshape((-1))
        for i in range(langs.shape[0])
    }
    return result


def load_files_and_langs(langs, paths, load_trees):
    paths = paths.split(',')
    if langs:
        langs = langs.split(',')

        assert len(langs) == len(paths)
        lang_labels = []
        tot_trees = []
        for lang, path in [(lang, path) for lang, path in zip(langs, paths) if lang and path]:
            trees = load_trees(path)
            tree_langs = [lang for i in range(len(trees))]
            tot_trees.extend(trees)
            lang_labels.extend(tree_langs)
        return (lang_labels, tot_trees)
    else:
        return (None, [tree for path in paths for tree in load_trees(path)])


def run_train(args):
    logger.addHandler(logging.FileHandler(f"{args.model_path_base}.log"))
    logger.info(args)
    if args.numpy_seed is not None:
        logger.info("Setting numpy random seed to {}...".format(
            args.numpy_seed))
        np.random.seed(args.numpy_seed)

    load_trees = trees.load_itg_trees if args.tree_type == 'itg' else trees.load_trees

    if args.language_embedding:
        language_embeddings = load_language_embeddings(args.language_embedding)
    else:
        language_embeddings = None

    train_langs, train_treebank = load_files_and_langs(
        args.train_langs, args.train_paths, load_trees)
    logger.info("Loaded {} training examples.".format(len(train_treebank)))

    dev_langs, dev_treebank = load_files_and_langs(args.dev_langs,
                                                   args.dev_paths, load_trees)

    if args.tree_type != 'treebank':
        dev_treebank = [tree.convert() for tree in dev_treebank]
    logger.info("Loaded {:,} development examples.".format(len(dev_treebank)))

    logger.info("Processing trees for training...")
    if args.tree_type == 'treebank':
        train_parse = [tree.convert() for tree in train_treebank]
    else:
        train_parse = train_treebank
    logger.info("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        logger.info("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    logger.info("Initializing model...")
    model = dy.ParameterCollection()
    if args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
            language_embeddings,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
            language_embeddings,
        )
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = args.checks_every
    best_dev_fscore = -np.inf
    best_processed = None
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_processed
        nonlocal total_processed

        dev_start_time = time.time()
        dev_predicted = []

        if args.dev_limit:
            dev_sample = np.random.choice(dev_treebank, args.dev_limit, replace=False)
        else:
            dev_sample = dev_treebank

        if language_embeddings:
            for lang, tree in zip(dev_langs, dev_sample):
                dy.renew_cg()
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                predicted, _ = parser.parse(sentence, lang=lang)
                dev_predicted.append(predicted.convert())
        else:
            for tree in dev_sample:
                dy.renew_cg()
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                predicted, _ = parser.parse(sentence)
                dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_sample,
                                    dev_predicted)

        dev_frs_score = evaluate.calc_frs(dev_sample, dev_predicted)
        logger.info("dev-fscore {} "
                    "dev-fuzzy reordering score {:4f} "
                    "dev-elapsed {} "
                    "total-elapsed {}".format(
                        dev_fscore,
                        dev_frs_score,
                        format_elapsed(dev_start_time),
                        format_elapsed(start_time),
                    ))

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        logger.info(
                            "Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_processed = total_processed
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            logger.info(
                "Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(
                0,
                min(args.train_limit, len(train_parse))
                if args.train_limit else len(train_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            if language_embeddings:
                for lang, tree in zip(
                        train_langs[start_index:start_index + args.batch_size],
                        train_parse[start_index:start_index +
                                    args.batch_size]):
                    sentence = [(leaf.tag, leaf.word)
                                for leaf in tree.leaves()]
                    if args.parser_type == "top-down":
                        _, loss = parser.parse(sentence, tree, args.explore,
                                               lang)
                    else:
                        _, loss = parser.parse(sentence, tree, lang)
                    batch_losses.append(loss)
                    total_processed += 1
                    current_processed += 1
            else:
                for tree in train_parse[start_index:start_index +
                                        args.batch_size]:
                    sentence = [(leaf.tag, leaf.word)
                                for leaf in tree.leaves()]
                    if args.parser_type == "top-down":
                        _, loss = parser.parse(
                            sentence, tree, explore=args.explore)
                    else:
                        _, loss = parser.parse(sentence, tree)
                    batch_losses.append(loss)
                    total_processed += 1
                    current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            logger.info("epoch {:,} "
                        "batch {:,}/{:,} "
                        "processed {:,} "
                        "batch-loss {:.4f} "
                        "epoch-elapsed {} "
                        "total-elapsed {}".format(
                            epoch,
                            start_index // args.batch_size + 1,
                            int(np.ceil(len(train_parse) / args.batch_size)),
                            total_processed,
                            batch_loss_value,
                            format_elapsed(epoch_start_time),
                            format_elapsed(start_time),
                        ))

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()
                if best_processed and total_processed - best_processed > args.patience:
                    break

        if best_processed and total_processed - best_processed > args.patience:
            logger.info(
                f"Patience limit of {args.patience} reached. Best processed: {best_processed}. Last epoch: {epoch - 1}"
            )
            break

def run_test(args):
    logger.info("Loading test trees from {}...".format(args.test_paths))

    test_langs, test_treebank = load_files_and_langs(
        args.test_langs, args.test_paths, trees.load_trees
        if args.tree_type == 'treebank' else trees.load_itg_trees)
    if args.tree_type != 'treebank':
        test_treebank = [tree.convert() for tree in test_treebank]

    logger.info("Loaded {:,} test examples.".format(len(test_treebank)))

    test_predicted = []
    if args.no_prediction:
        for tree in test_treebank:
            children = [trees.LeafTreebankNode(leaf.tag, leaf.word) for leaf in tree.leaves()]
            test_predicted.append(trees.InternalTreebankNode('S', children))
        start_time = time.time()
    else:
        logger.info("Loading model from {}...".format(args.model_path_base))
        model = dy.ParameterCollection()
        [parser] = dy.load(args.model_path_base, model)

        if args.language_embedding and test_langs:
            logger.debug(
                f"Setting up language embeddings from {args.language_embedding} for {test_langs}"
            )
            parser.lang_embeddings = load_language_embeddings(
                args.language_embedding)

        logger.info("Parsing test sentences...")

        start_time = time.time()


        if test_langs and args.language_embedding:
            for lang, tree in zip(test_langs, test_treebank):
                dy.renew_cg()
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                predicted, _ = parser.parse(sentence, lang=lang)
                test_predicted.append(predicted.convert())
        else:
            for tree in test_treebank:
                dy.renew_cg()
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                predicted, _ = parser.parse(sentence)
                test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)
    test_frs_score = evaluate.calc_frs(test_treebank, test_predicted)
    logger.info("test-fscore {} "
                "test-fuzzy-reordering-score {:4f} "
                "test-elapsed {}".format(test_fscore, test_frs_score,
                                         format_elapsed(start_time)))
    print(f"Printing to this path {args.result_path}")
    with open(args.result_path, 'w') as out_file:
        result_dict = {
        'recall': test_fscore.recall,
            'precision': test_fscore.precision,
            'fscore': test_fscore.fscore,
            'fuzzy_reorder_score': test_frs_score
        }
        yaml.dump(result_dict, out_file)


def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)

    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)

    # Model options
    subparser.add_argument(
        "--parser-type", choices=["top-down", "chart"], required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--model-path-base", required=True)
    '''
    Language embedding file should look something along the lines of the formatting
    done with the uriel data
    '''
    subparser.add_argument("--language-embedding", type=str)

    # Paths for eval and training
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-paths", default="data/02-21.10way.clean")
    subparser.add_argument("--train-langs", default=None)
    subparser.add_argument("--train-limit", type=int, default=None)
    subparser.add_argument("--dev-paths", default="data/22.auto.clean")
    subparser.add_argument("--dev-langs", default=None)
    subparser.add_argument("--dev-limit", type=int, default=None)

    # Training options
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--patience", type=int, default=5000)
    # Checks every x number of batches
    subparser.add_argument("--checks-every", type=int, default=1000)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument(
        "--tree-type", choices=["itg", "treebank"], required=True)


    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)

    for arg in dynet_args:
        subparser.add_argument(arg)

    subparser.add_argument("--no-prediction", action='store_true')
    subparser.add_argument("--model-path-base")
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-paths", default="data/23.auto.clean")
    subparser.add_argument("--test-langs", type=str)
    subparser.add_argument(
        "--tree-type", choices=["itg", "treebank"], required=True)
    subparser.add_argument("--language-embedding", type=str)
    subparser.add_argument('--result-path', type=str, required=True)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()

from typing import List
from itertools import groupby, product
import math
import os.path
import re
import subprocess
import tempfile

from trees import InternalTreebankNode, LeafTreebankNode, TreebankNode


class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f})".format(
            self.recall, self.precision, self.fscore)


def permute_from_tree(tree: TreebankNode) -> List[str]:
    if isinstance(tree, LeafTreebankNode):
        return [tree.word]
    # Assume it's an InternalTreebankNode
    elif tree.label == 'R':
        return [
            word for child in list(tree.children)[::-1]
            for word in permute_from_tree(child)
        ]
    # Straight label
    else:
        return [
            word for child in list(tree.children)
            for word in permute_from_tree(child)
        ]


'''Calculate the "fuzzy reordering score" from "A Lightweight Evaluation
Framework for Machine Translation Reordering" (Talbot 2011)'''


def calc_frs(gold_trees: List[TreebankNode],
             predicted_trees: List[TreebankNode]) -> float:
    result = []
    total_len = 0
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        gold_perm = permute_from_tree(gold_tree)
        if len(gold_perm) > 0:
            predicted_perm = permute_from_tree(predicted_tree)
            gold_indices = {
                word: {idx
                       for idx, _ in items}
                for word, items in groupby(
                    sorted(list(enumerate(gold_perm)), key=lambda x: x[1]),
                    key=lambda x: x[1])
            }
            idx_mismatches = 0
            # Compare each word w/ prev_word to see if it's offset by 1
            for idx, word in enumerate(predicted_perm[1:]):
                prev_word = predicted_perm[idx]
                if 1 not in {
                        y - x
                        for x, y in product(gold_indices[prev_word],
                                            gold_indices[word]) if y - x > 0
                }:
                    idx_mismatches += 1

            # Explicit formulation for score
            score = 1 - idx_mismatches / (len(gold_perm) - 1)

            result.append((len(gold_perm) - 1) * score)
            total_len += len(gold_perm) - 1
    return sum(result) / total_len


def evalb(evalb_dir, gold_trees, predicted_trees):
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        assert isinstance(gold_tree, TreebankNode)
        assert isinstance(predicted_tree, TreebankNode)
        gold_leaves = list(gold_tree.leaves())
        predicted_leaves = list(predicted_tree.leaves())
        assert len(gold_leaves) == len(predicted_leaves)
        assert all(gold_leaf.word == predicted_leaf.word
                   for gold_leaf, predicted_leaf in zip(
                       gold_leaves, predicted_leaves))

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        for tree in gold_trees:
            outfile.write("{}\n".format(tree.linearize()))

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(tree.linearize()))

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    fscore = FScore(math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
                break

    success = (not math.isnan(fscore.fscore) or fscore.recall == 0.0
               or fscore.precision == 0.0)

    if success:
        temp_dir.cleanup()
    else:
        print("Error reading EVALB results.")
        print("Gold path: {}".format(gold_path))
        print("Predicted path: {}".format(predicted_path))
        print("Output path: {}".format(output_path))

    return fscore

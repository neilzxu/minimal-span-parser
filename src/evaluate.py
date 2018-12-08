from typing import List
from collections import Counter
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


def fuzzy_reordering_score(perm_ref, perm_sys):
    print(perm_ref)
    print(perm_sys)
    assert (Counter(perm_ref) == Counter(perm_sys))
    words = set(perm_sys)
    alignments = []
    for word in words:
        ref_indices = sorted(
            [idx for (idx, token) in enumerate(perm_ref) if token == word])
        sys_indices = sorted(
            [idx for (idx, token) in enumerate(perm_sys) if token == word])
        alignments.extend(list(zip(sys_indices, ref_indices)))
    chunk_indices = [ref_index for _, ref_index in sorted(alignments)]

    chunk_ct = 1 + len([
        idx for idx in range(len(chunk_indices) - 1)
        if chunk_indices[idx + 1] - chunk_indices[idx] != 1
    ])

    return 1 - (chunk_ct - 1) / (len(perm_ref))


def calc_frs(gold_trees: List[TreebankNode],
             predicted_trees: List[TreebankNode]) -> float:
    result = []
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        gold_perm = ' '.join(permute_from_tree(gold_tree))
        if len(gold_perm) > 1:
            predicted_perm = ' '.join(permute_from_tree(predicted_tree))
            result.append(fuzzy_reordering_score(gold_perm, predicted_perm))

    return sum(result) / len(result)


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

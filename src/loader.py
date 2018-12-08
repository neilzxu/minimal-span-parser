from typing import Dict, Tuple, List, Union

from trees import TreebankNode, load_itg_trees, load_trees

import numpy as np


class Loader:
    tree_load_dict = {
        'itg': load_itg_trees,
        'treebank': load_trees,
    }

    def __init__(self, cmd_args):
        self.tree_type = cmd_args.tree_type
        self.load_trees = Loader.tree_load_dict[self.tree_type]

    def load_lang_emb(self, lang_emb_path) -> Dict[str, np.ndarray]:
        arr = np.load(lang_emb_path)
        langs = arr['langs']
        features = arr['data']

        result = {
            langs[i][:2]: features[i].reshape((-1))
            for i in range(langs.shape[0])
        }
        return result

    def load_treebank(self, lang_list, path_list
                      ) -> Tuple[Union[None, List[str]], List[TreebankNode]]:
        lang_labels = []
        tot_trees = []
        if lang_list:
            assert len(lang_list) == len(path_list)
            for lang, path in [(lang, path)
                               for lang, path in zip(lang_list, path_list)
                               if lang and path]:
                trees = self.load_trees(path)
                tree_langs = [lang for i in range(len(trees))]
                tot_trees.extend(trees)
                lang_labels.extend(tree_langs)
            return lang_labels, tot_trees
        else:
            return (None, [
                tree for path in path_list for tree in self.load_trees(path)
            ])

    def load_parse(self, lang_list, path_list):
        langs, trees = self.load_treebank(lang_list, path_list)
        return langs, [tree.convert() for tree in trees]

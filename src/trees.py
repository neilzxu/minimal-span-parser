import collections.abc
import re


class TreebankNode(object):
    pass


class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(tree.children[0],
                                                     InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

    def __repr__(self):
        if all(isinstance(child, LeafTreebankNode) for child in self.children):
            children = ' '.join([child.word for child in self.children])
        else:
            children = ' '.join([child.__repr__() for child in self.children])
        return f"({self.label} {children})"


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

    def __repr__(self):
        return f"({self.tag} {self.word})"


class ParseNode(object):
    pass


class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(left.right == right.left
                   for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

    def __repr__(self):
        if all(isinstance(child, LeafParseNode) for child in self.children):
            children = ' '.join([child.word for child in self.children])
        else:
            children = ' '.join([child.__repr__() for child in self.children])
        return f"({self.label} {children})"


class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def __repr__(self):
        return f"({self.tag} {self.word})"


def load_itg_tree(sent, tree, idx=None):
    sent_tokens = sent.strip().split()
    ctx_stack = []
    ctx = []
    cur_label = None
    for token in tree:
        if token == '(' and token != None:
            ctx_stack.append((cur_label, ctx))
            ctx = []
        elif token == 'R' or token == 'S':
            assert not ctx
            cur_label = token
        elif re.match(r'\[([0-9]+)-([0-9]+)\]', token):
            res = re.match(r'\[([0-9]+)-([0-9]+)\]', token)
            start, end = int(res.group(1)), int(res.group(2))
            if start < 0 or end >= len(sent_tokens):
                raise ValueError(
                    f"ITG idx ({idx}) invalid. expected: {len(sent_tokens)}, actual: [{start}-{end}]"
                )
            ctx.extend([
                LeafTreebankNode('S', sent_tokens[idx])
                for idx in range(start, end + 1)
            ])
        elif re.match(r'[0-9]+', token):
            ctx.append(LeafTreebankNode('S', sent_tokens[int(token)]))
        elif token == ')':
            tree = InternalTreebankNode(cur_label, ctx)
            if ctx_stack:
                cur_label, ctx = ctx_stack.pop()
                ctx.append(tree)
            else:
                return tree
        else:
            raise ValueError(f"Invalid token {token} in ITG tree")
    assert len(ctx) == 1
    return ctx[0]


def load_itg_trees(path):
    with open(path, encoding='utf-8') as in_file:
        langs = []
        res = []
        for idx, sent, tree in zip(in_file, in_file, in_file):
            if tree.strip() != 'None':
                tokens = tree.replace("(", " ( ").replace(")", " ) ").split()
                res.append(load_itg_tree(sent, tokens, idx))
    return res


def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees

from collections import Counter
from operator import attrgetter
from typing import TypeVar

from smallgraphlib.custom_types import Node


Self = TypeVar("Self")


class Tree:
    def __init__(self, root: Node, *branches: Self) -> None:
        self.root = root
        self.branches = branches


class HuffmanTree(Tree):
    def __init__(self, *branches: "HuffmanTree", char: str = None, weight: int = None) -> None:
        if len(branches) == 2:
            if char is not None or weight is not None:
                raise ValueError("Char and weight can't be set, except for leaves.")
        elif len(branches) == 0:
            if char is None or weight is None:
                raise ValueError("Char and weight must be set for leaves.")
        else:
            raise ValueError(f"There must be either 0 or 2 branches, not {len(branches)}.")
        char = min(branch.char for branch in branches) if branches else char
        weight = sum(branch.weight for branch in branches) if branches else weight
        super().__init__((weight, char), *branches)

    @classmethod
    def from_text(cls, text: str) -> "HuffmanTree":
        trees = {
            HuffmanTree(char=letter, weight=occurrences) for letter, occurrences in Counter(text).items()
        }
        sort_func = attrgetter("root")
        while len(trees) >= 2:
            print([tree.root for tree in trees])
            # Select the two smallest values
            tree1 = min(trees, key=sort_func)
            trees.remove(tree1)
            tree2 = min(trees, key=sort_func)
            trees.remove(tree2)
            trees.add(HuffmanTree(tree1, tree2))
        assert len(trees) == 1
        return trees.pop()

    @property
    def weight(self):
        return self.root[0]

    @property
    def char(self):
        return self.root[1]

    @property
    def is_leaf(self) -> bool:
        return len(self.branches) == 0

    @property
    def left_branch(self) -> "HuffmanTree":
        return self.branches[0]

    @property
    def right_branch(self) -> "HuffmanTree":
        return self.branches[1]

    def compress(self, text: str) -> bytes:
        compressed = []
        buffer = ""
        for char in text:
            buffer += self.labels[char]
            while len(buffer) >= 8:
                compressed.append(sum(2**i * int(c) for i, c in enumerate(buffer[:8])))
                buffer = buffer[8:]
        # On finit de vider le buffer.
        # Cela revient à ajouter des bits nuls, puisque le nombre de bits final doit
        # être un multiple de 8 (on ne peut renvoyer que des octets complets).
        # Si on voulait créer un logiciel de compression réellement utilisable, il faudrait donc
        # soit avoir un caractère de fin de chaîne, soit préciser le nombre de bits finaux
        # inutiles (entre 0 et 7).
        # On pourrait par exemple décider que par convention les 3 premiers bits servent à encoder le
        # nombre de bits finaux inutiles.
        # Accessoirement, il faudrait aussi que la chaîne de caractères intègre le dictionnaire de compression
        # en début de chaîne, et donc se mettre d'accord sur un format, etc.
        if buffer:
            compressed.append(sum(2**i * int(c) for i, c in enumerate(buffer[:8])))
        return bytes(compressed)

    def uncompress(self, bits: bytes) -> str:
        position = self
        read: list[str] = []
        for byte in bits:
            for i in range(8):
                digit, byte = byte % 2, byte // 2
                position = position.branches[digit]
                if position.is_leaf:
                    read.append(position.char)
                    position = self
        return "".join(read)

    def decode(self, bits: str) -> str:
        position = self
        read: list[str] = []
        for digit in bits:
            position = position.branches[int(digit)]
            if position.is_leaf:
                read.append(position.char)
                position = self
        return "".join(read)

    def encode(self, text: str) -> str:
        return "".join(self.labels[char] for char in text)

    @property
    def labels(self) -> dict[str, str]:
        """Binary labels."""
        if self.is_leaf:
            return {self.char: ""}
        return {
            key: str(i) + value
            for i, branch in enumerate(self.branches)
            for key, value in branch.labels.items()
        }

    def __repr__(self):
        if self.is_leaf:
            return f"HuffmanTree(char={self.char}, weight={self.weight})"
        return f"HuffmanTree({self.left_branch!r}, {self.right_branch!r})"

    def __str__(self):
        lines = []
        shift = len(str(self.weight)) + 1
        for n, line in enumerate(str(self.left_branch).split("\n")):
            if n == 0:
                # lines.append("\u252C\u2500\u2500" + line)
                lines.append(f"{self.weight}\u2500\u2500" + line)
            else:
                lines.append("\u2502" + shift * " " + line)
        for n, line in enumerate(str(self.right_branch).split("\n")):
            if n == 0:
                lines.append("\u2514" + shift * "\u2500" + line)
            else:
                lines.append((shift + 1) * " " + line)
        return "\n".join(lines)


def encode(text: str) -> str:
    """Return a string of `0` and `1` representing the bits of a text encoded using Huffman algorithm."""
    tree = HuffmanTree.from_text(text)
    return tree.encode(text)

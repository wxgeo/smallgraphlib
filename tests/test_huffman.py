from smallgraphlib.huffman import encode, HuffmanTree


def test_compress():
    text = "MAELLE ALLUMA LA LAMPE."
    compressed = "101001001111100010001111011010100010110001011001010111110001110"
    tree = HuffmanTree.from_text(text)
    print(tree.labels)
    assert tree.encode(text) == compressed
    assert tree.decode(compressed) == text
    assert encode(text) == compressed


def test_str():
    tree = HuffmanTree.from_text("CENT SCIES SCIERENT.")
    assert tree.as_dict() == {
        (20, " "): [(8, "E"), (12, " ")],
        (8, "E"): [(4, "E"), (4, "I")],
        (4, "E"): [],
        (4, "I"): [(2, "I"), (2, "N")],
        (2, "I"): [],
        (2, "N"): [],
        (12, " "): [(5, "C"), (7, " ")],
        (5, "C"): [(2, "T"), (3, "C")],
        (2, "T"): [],
        (3, "C"): [],
        (7, " "): [(3, "S"), (4, " ")],
        (3, "S"): [],
        (4, " "): [(2, " "), (2, ".")],
        (2, " "): [],
        (2, "."): [(1, "."), (1, "R")],
        (1, "."): [],
        (1, "R"): [],
    }
    assert eval(repr(tree)).as_dict() == tree.as_dict()
    assert (
        str(tree)
        == """20──8──(E)
│   └──4──(I)
│      └──(N)
└───12──5──(T)
    │   └──(C)
    └───7──(S)
        └──4──( )
           └──2──(.)
              └──(R)"""
    )

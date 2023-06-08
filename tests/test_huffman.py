from smallgraphlib.huffman import encode, HuffmanTree


def test_compress():
    text = "MAELLE ALLUMA LA LAMPE."
    compressed = "101001001111100010001111011010100010110001011001010111110001110"
    tree = HuffmanTree.from_text(text)
    print(tree.labels)
    assert tree.encode(text) == compressed
    assert tree.decode(compressed) == text
    assert encode(text) == compressed

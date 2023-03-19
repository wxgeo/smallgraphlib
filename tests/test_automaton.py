from smallgraphlib import Acceptor, DeterministicTransducer


def test_Acceptor_from_string():
    g1 = Acceptor.from_string(">(I):a|b--1 / (1):a--2;b--3 / (2):a--1|I / 3")
    assert g1.alphabet == ("a", "b")
    assert set(g1.states) == {"1", "2", "3", "I"}
    assert g1.transitions == (
        ("1", "2", "a"),
        ("1", "3", "b"),
        ("2", "1", "a"),
        ("2", "I", "a"),
        ("I", "1", "a"),
        ("I", "1", "b"),
    )
    assert g1.is_directed
    assert g1.initial_states == {"I"}
    assert g1.final_states == {"I", "1", "2"}
    tikz1 = g1.as_tikz()
    print(tikz1)
    g2 = Acceptor.from_string(">(I):a,b:1 ; (1):a:2+b:3 ; (2):a:1,I ; 3", sep=(";", ":", "+", ":", ","))
    assert g2 == g1
    g3 = Acceptor.from_string("(I):a,b:1 ; >(1):a:2+b:3 ; (2):a:1,I ; 3", sep=(";", ":", "+", ":", ","))
    assert g3 != g1
    g4 = Acceptor.from_string(">(I):a,b:1 ; 1:a:2+b:3 ; (2):a:1,I ; 3", sep=(";", ":", "+", ":", ","))
    assert g4 != g1


def test_Acceptor_deterministic():
    g = Acceptor(
        (1, 2, 3),
        (1, 1, "0"),
        (1, 2, "1"),
        (2, 2, "0"),
        (2, 3, "1"),
        (3, 3, "0"),
        (3, 1, "1"),
        alphabet="01",
        initial_states=(1,),
        final_states=(2,),
    )
    assert g.is_deterministic
    g = Acceptor(
        (1, 2, 3),
        (1, 1, "0"),
        (1, 1, "1"),
        (1, 2, "1"),
        (2, 2, "0"),
        (2, 3, "1"),
        (3, 3, "0"),
        (3, 1, "1"),
        alphabet="01",
        initial_states=(1,),
        final_states=(2,),
    )
    assert g.transition_func(1, "1") == {1, 2}
    assert not g.is_deterministic
    g = Acceptor(
        (1, 2, 3),
        (1, 1, "0"),
        (1, 2, "1"),
        (2, 2, "0"),
        (2, 3, "1"),
        (3, 3, "0"),
        alphabet="01",
        initial_states=(1,),
        final_states=(2,),
    )
    assert g.transition_func(3, "1") == set()
    assert not g.is_deterministic


def test_Acceptor_recognize():
    # g recognize all the binary words for which the number of 1 is a multiple of 3
    g = Acceptor(
        (1, 2, 3),
        (1, 1, "0"),
        (1, 2, "1"),
        (2, 2, "0"),
        (2, 3, "1"),
        (3, 3, "0"),
        (3, 1, "1"),
        alphabet="01",
        initial_states=(1,),
        final_states=(1,),
    )
    assert g.recognize("")
    assert g.recognize(5 * "0")
    assert g.recognize(9 * "1")
    assert not g.recognize(7 * "1")
    word = "100101000101000100"
    assert word.count("1") % 3 == 0
    assert g.recognize(word)
    word = "10010000101000100"
    assert word.count("1") % 3 != 0
    assert not g.recognize(word)


def test_Automaton_repr_eq():
    g = Acceptor.from_string(">I:1;0--1 / (1):1;0--I")
    assert eval(repr(g)) == g


def test_Acceptor_alphabet_name():
    g1 = Acceptor.from_string(">I:a--1;b / (1):a|b--I")
    g2 = Acceptor.from_string(">I:a--1;b / (1):**--I")
    g3 = Acceptor.from_string(">I:a--1;b / (1):A--I", alphabet_name="A")
    g4 = Acceptor.from_string(r">I:a--1;b / (1):\Sigma--I", alphabet_name=r"\Sigma")
    assert g2 == g1
    assert g3 == g1
    assert g4 == g1
    assert g1._tikz_labels("1", "I") == [r"$a$,$b$"]
    assert g2._tikz_labels("1", "I") == [r"$a$,$b$"]
    assert g3._tikz_labels("1", "I") == [r"$A$"]
    assert g4._tikz_labels("1", "I") == [r"$\Sigma$"]


def test_DeterministicTransducer():
    # This automaton count the number of "ba" substrings.
    g = DeterministicTransducer(
        ("I", "B"),
        ("I", "I", "a"),
        ("I", "B", "b"),
        ("B", "B", "b"),
        ("B", "I", "a", "*"),
        input_alphabet="ab",
        output_alphabet="*",
        initial_state="I",
    )
    substrings = ["aaa", "ba", "bb", "ba", "ba", "bb", "ba", "aa", "ba", "ba", "a"]
    assert all(s == "ba" or "ba" not in s for s in substrings)
    # The output must have one star for each "ba" substring.
    assert g.translate("".join(substrings)) == substrings.count("ba") * "*"

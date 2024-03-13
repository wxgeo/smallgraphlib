from smallgraphlib.flow import FlowNetwork


def test_FlowNetwork():
    f = FlowNetwork.from_dict(SA=10, AC=2, SB=3, AB=4, BD=5, DC=3, CP=6, DP=4)
    assert f.find_path(f.source, f.sink) in (["S", "B", "D", "P"], ["S", "A", "C", "P"])
    assert f.get_max_flow() == FlowNetwork(
        ("A", "B", "C", "D", "P", "S"),
        ("A", "B", 2),
        ("A", "C", 2),
        ("B", "D", 5),
        ("C", "P", 3),
        ("D", "C", 1),
        ("D", "P", 4),
        ("S", "A", 4),
        ("S", "B", 3),
    )
    assert f.get_max_flow_value() == 7

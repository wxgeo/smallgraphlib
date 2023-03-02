import re
from typing import Iterable

from smallgraphlib.labeled_graphs import LabeledEdge, LabeledDirectedGraph, Label
from smallgraphlib.core import Node


class UnknownState(RuntimeError):
    """Error raised when a state is unknown to the automat."""


class UnknownLetter(RuntimeError):
    """Error raised when a letter is unknown to the automat."""


class Automaton(LabeledDirectedGraph):
    def __init__(
        self,
        states: Iterable[Node],
        *transitions: LabeledEdge,
        alphabet: Iterable[str] | str,
        initial_states: Iterable[Node],
        final_states: Iterable[Node],
        sort_nodes: bool = True,
    ):
        states = tuple(states)
        final_states = tuple(final_states)
        alphabet = tuple(alphabet)
        for state in initial_states:
            if state not in states:
                raise UnknownState(f"Initial state {state} must be one of the automaton states: {states}.")
        for state in final_states:
            if state not in states:
                raise UnknownState(f"Final state {state} must be one of the automaton states: {states}.")
        for _, _, label in transitions:
            if label not in alphabet:
                raise UnknownLetter(f"Letter {label} must be one of the automat alphabet: {alphabet}.")
        super().__init__(states, *transitions, sort_nodes=sort_nodes)
        self.alphabet = tuple(sorted(alphabet))
        self.initial_states = frozenset(initial_states)
        self.final_states = frozenset(final_states)

    @property
    def states(self) -> tuple[Node, ...]:
        return self.nodes

    @property
    def transitions(self) -> tuple[LabeledEdge, ...]:
        return self.labeled_edges

    def _tikz_specific_node_style(self, node: Node) -> str:
        styles = []
        if node in self.final_states:
            styles.append("double,fill=lightgray")
        if node in self.initial_states:
            styles.append("rectangle")
        return ",".join(styles)

    def _tikz_labels(self, node1, node2) -> list[str]:
        def latex(letter: str) -> str:
            return f"${letter}$" if letter else r"$\varepsilon$"

        labels = self.labels(node1, node2)
        return [",".join(latex(label) for label in labels)] if labels else []

    def _tikz_count_edges(self, node1: Node, node2: Node) -> int:
        """For automatons, any parallel oriented edges are displayed as a unique oriented edge,
        with merged labels (for example, two edges labeled `a` and `b` are replaced by a unique edge
        labeled `a,b`).
        """
        return min(self.count_edges(node1, node2), 1)

    @staticmethod
    def _parse_transitions(substr: str) -> list[tuple[Node | None, Label]]:
        state_transitions: list[tuple[Node | None, Label]] = []
        for transition in substr.split("&"):
            match transition.split(":"):
                case [letters, next_states]:
                    state_transitions.extend(
                        (next_state.strip(), letter.strip())
                        for next_state in next_states.split(",")
                        for letter in letters.split(",")
                    )
                case [letters]:
                    state_transitions.extend((None, letter.strip()) for letter in letters.split(","))
                case _:
                    raise ValueError(f"Invalid format for {transition!r}.")
        return state_transitions

    @classmethod
    def from_string(cls, string: str):
        """Constructor used to generate an automaton from a string.

        `Automaton.from_string(">(I):a,b:1 ; (1):a:2&b:3 ; (2):a:1,I ; 3")`
        will generate an automaton of 4 states: `I`, `1`, `2` and `3`,
        each state information being separated by `;`.

        Each state can be marked as initial or final:

            - The parentheses `()` around states `I`, `1` and `2` mean those states will be final.
            - The `>` before state `I` means it's the initial state.

        After the state name, `:` will introduce the eventual transitions:

            - `I:a,b:1` means that reading `a` or `b` while being in state `I` leads to state `1`.
            - `1:a:2&b:3` means that in state `1`, reading `a` leads to state `2`
              and reading `b` leads to state `3`.
            - `2:a:1,I` means that in state `2`, reading `a` leads either to state `1` or to state `I`.

        If the letter is left empty, like in `2::1`, an epsilon-transition is assumed
        (transition without reading any letter).
        """
        all_states = []
        final_states = []
        initial_states = []
        transitions = []
        alphabet = set()
        for state_info in string.split(";"):
            match state_info.strip().split(":", maxsplit=1):
                case ["", *_]:
                    raise ValueError(f"Empty state name in {state_info!r}.")
                case [state, transitions_str]:
                    transitions_data: list[tuple[Node | None, str]] = cls._parse_transitions(transitions_str)
                case [state]:
                    transitions_data = []
                case _:
                    raise ValueError(f"Invalid format for {state_info!r}.")
            # Parse state format
            state = state.strip()
            initial = final = False
            if state[0] == ">":
                initial = True
                state = state[1:]
            if m := re.match(r"\((.+)\)", state):
                final = True
                state = m.group(1)
            # Collect all data
            all_states.append(state)
            if final:
                final_states.append(state)
            if initial:
                initial_states.append(state)
            for next_state, letter in transitions_data:
                if next_state is None:
                    next_state = state
                transitions.append((state, next_state, letter))
                alphabet.add(letter)
        return cls(
            all_states,
            *transitions,
            alphabet=alphabet,
            initial_states=initial_states,
            final_states=final_states,
        )

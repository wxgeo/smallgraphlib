import re
from abc import ABC
from typing import Iterable, Generic, Type, TypeVar, NewType, cast

from smallgraphlib.core import Node
from smallgraphlib.labeled_graphs import LabeledEdge, LabeledDirectedGraph, Label
from smallgraphlib.utilities import cached_property

_T = TypeVar("_T")
Letter = NewType("Letter", str)


class UnknownState(RuntimeError):
    """Error raised when a state is unknown to the automat."""


class UnknownLetter(RuntimeError):
    """Error raised when a letter is unknown to the automat."""


class Automaton(LabeledDirectedGraph, ABC, Generic[Node]):
    def __init__(
        self,
        states: Iterable[Node],
        *transitions: LabeledEdge,
        alphabet: Iterable[Letter] | str,
        initial_states: Iterable[Node],
        alphabet_name: str = None,
        sort_nodes: bool = True,
    ):
        states = tuple(states)
        alphabet = cast(Iterable[Letter], alphabet)
        sorted_alphabet: tuple[Letter, ...] = tuple(sorted(alphabet))
        for letter in sorted_alphabet:
            if len(letter) != 1:
                raise ValueError(
                    f"Invalid value for letter: {letter!r}. Letters must be strings of length 1."
                )
        for state in initial_states:
            if state not in states:
                raise UnknownState(f"Initial state {state} must be one of the automaton states: {states}.")
        self._transitions_dict: dict[tuple[Node, Letter], set[Node]] = {}
        for state1, state2, label in transitions:
            # Label must be either a letter of the alphabet or the empty word.
            if label and label not in alphabet:
                raise UnknownLetter(f"Letter {label} must be in the automat alphabet: {alphabet}.")
            self._transitions_dict.setdefault((state1, label), set()).add(state2)
        super().__init__(states, *transitions, sort_nodes=sort_nodes)
        self.alphabet = sorted_alphabet
        self.initial_states = frozenset(initial_states)
        self.alphabet_name = alphabet_name

    @property
    def states(self) -> tuple[Node, ...]:
        return self.nodes

    @cached_property
    def is_deterministic(self):
        if len(self.initial_states) > 1:
            return False
        for state in self.nodes:
            for letter in self.alphabet:
                if len(self._transitions_dict.get((state, letter), set())) != 1:
                    return False
        return True

    @classmethod
    def from_string(
        cls: Type[_T],
        string: str,
        sep: tuple[str, str, str, str] = ("/", ";", "--", "|"),
        alphabet_name: str = None,
    ) -> _T:
        """Constructor used to generate an automaton from a string.

            >>> Automaton.from_string(">(I)--a|b--1  /  (1)--a--2;b--3  /  (2)--a--1|I  /  3")

        will generate an automaton of 4 states: `I`, `1`, `2` and `3`,
        each state information being separated by `/`.

        Each state can be marked as initial or final:

            - The parentheses `()` around states `I`, `1` and `2` mean those states will be final.
            - The `>` before state `I` means it's the initial state.

        After the state name, `--` will introduce the eventual transitions:

            - `I--a|b--1` means that reading `a` or `b` while being in state `I` leads to state `1`.
            - `1--a--2|b--3` means that in state `1`, reading `a` leads to state `2`
              and reading `b` leads to state `3`.
            - `2--a--1|I` means that in state `2`, reading `a` leads either to state `1` or to state `I`.

        If the letter is left empty, like in `2-- --1`, an epsilon-transition is assumed
        (transition without reading any letter).

        Separators may be changed using `sep` keyword:

            >>> Automaton.from_string(">(I):a,b:1 ; (1):a:2+b:3 ; (2):a:1,I ; 3", sep=(";", "+", ":", ","))

        If a transition applies for every letter, one may use the alphabet name or `**` instead of listing
        all the letters.

            >>> Automaton.from_string(">I--a--1;b / (1)--**--I")
        """

        def parse_transitions(substr: str) -> list[tuple[str | None, Letter]]:
            state_transitions: list[tuple[str | None, Letter]] = []
            for transition in substr.split(sep[1]):
                match transition.split(sep[2]):
                    case [letters, next_states]:
                        state_transitions.extend(
                            (next_one.strip(), Letter(_letter.strip()))
                            for next_one in next_states.split(sep[3])
                            for _letter in letters.split(sep[3])
                        )
                    case [letters]:
                        state_transitions.extend(
                            (None, Letter(_letter.strip())) for _letter in letters.split(sep[3])
                        )
                    case _:
                        raise ValueError(f"Invalid format for {transition!r}.")
            return state_transitions

        all_states: list[str] = []
        final_states: list[str] = []
        initial_states: list[str] = []
        transitions: list[tuple[str, str, Letter]] = []
        alphabet: set[Letter] = set()
        state: str
        next_state: str
        transitions_str: str
        for state_info in string.split(sep[0]):
            match state_info.strip().split(sep[2], maxsplit=1):
                case ["", *_]:
                    raise ValueError(f"Empty state name in {state_info!r}.")
                case [state, transitions_str]:
                    transitions_data: list[tuple[str | None, Letter]] = parse_transitions(transitions_str)
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
            for _next_state, letter in transitions_data:
                next_state = state if _next_state is None else _next_state
                transitions.append((state, next_state, letter))
                alphabet.add(letter)

        # `**` notation stands for all alphabet's letters.
        alphabet -= {"**", alphabet_name}
        updated_transitions: list[tuple[str, str, str]] = []
        for state, next_state, letter in transitions:
            if letter == alphabet_name or letter == "**":
                updated_transitions.extend((state, next_state, alpha) for alpha in alphabet)
            else:
                updated_transitions.append((state, next_state, letter))

        return Acceptor[str, str](
            all_states,
            *updated_transitions,
            alphabet=alphabet,
            initial_states=initial_states,  # type: ignore
            final_states=final_states,  # type: ignore
            alphabet_name=alphabet_name,
        )


class Acceptor(Automaton, Generic[Node, Label]):
    def __init__(
        self,
        states: Iterable[Node],
        *transitions: LabeledEdge,
        alphabet: Iterable[Letter] | str,
        initial_states: Iterable[Node],
        final_states: Iterable[Node],
        alphabet_name: str = None,
        sort_nodes: bool = True,
    ):
        alphabet = cast(Iterable[Letter], alphabet)
        super().__init__(
            states,
            *transitions,
            alphabet=alphabet,
            initial_states=initial_states,
            alphabet_name=alphabet_name,
            sort_nodes=sort_nodes,
        )
        final_states = tuple(final_states)
        for state in final_states:
            if state not in states:
                raise UnknownState(f"Final state {state} must be one of the automaton states: {states}.")
        self.final_states = frozenset(final_states)

    @cached_property
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

        labels = sorted(self.labels(node1, node2))
        if self.alphabet_name is not None and sorted(labels) == list(self.alphabet):
            return [latex(self.alphabet_name)]
        return [",".join(latex(label) for label in labels)] if labels else []

    def _tikz_count_edges(self, node1: Node, node2: Node) -> int:
        """For automatons, any parallel oriented edges are displayed as a unique oriented edge,
        with merged labels (for example, two edges labeled `a` and `b` are replaced by a unique edge
        labeled `a,b`).
        """
        return min(self.count_edges(node1, node2), 1)

    def transition_func(self, state: Node, letter: Letter) -> frozenset[Node]:
        return frozenset(self._transitions_dict.get((state, letter), ()))

    def recognize(self, word: str, _start: Iterable[Node] = None) -> bool:
        states = set(_start) if _start is not None else self.initial_states
        if word == "":
            # Any of the current states must be final.
            return len(states & self.final_states) != 0
        return any(self.recognize(word[1:], self.transition_func(state, Letter(word[0]))) for state in states)

    def __eq__(self, other):
        return (
            isinstance(other, Acceptor)
            and other.alphabet == self.alphabet
            and other.states == self.states
            and other.initial_states == self.initial_states
            and other.final_states == self.final_states
            and all(
                other.transition_func(state, letter) == self.transition_func(state, letter)
                for letter in self.alphabet
                for state in self.states
            )
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.states!r}, "
            f"{', '.join(repr(transition) for transition in self.transitions)}, "
            f"alphabet={''.join(self.alphabet)!r}, initial_states={set(self.initial_states)!r}, "
            f"final_states={set(self.final_states)!r}, alphabet_name={self.alphabet_name!r})"
        )


class DeterministicTransducer(Automaton, Generic[Node]):
    def __init__(
        self,
        states: Iterable[Node],
        *transitions: tuple[Node, Node, Letter] | tuple[Node, Node, Letter, str],
        input_alphabet: Iterable[str] | str,
        output_alphabet: Iterable[str] | str,
        initial_states: Iterable[Node],
        alphabet_name: str = None,
        sort_nodes: bool = True,
    ):
        input_alphabet = cast(Iterable[Letter], input_alphabet)
        self._outputs_dict: dict[tuple[Node, Letter], str] = {}
        output_free_transitions = []
        for transition in transitions:
            match transition:
                case state1, state2, input_letter, output_message:
                    pass
                case state1, state2, input_letter:
                    output_message = ""
                case _:
                    raise ValueError(f"Invalid format for transition {transition!r}.")
            self._outputs_dict[(state1, input_letter)] = output_message  # type: ignore
            output_free_transitions.append((state1, state2, input_letter))
        super().__init__(
            states,
            *output_free_transitions,
            alphabet=input_alphabet,
            initial_states=initial_states,
            alphabet_name=alphabet_name,
            sort_nodes=sort_nodes,
        )
        sorted_output_alphabet: tuple[str, ...] = tuple(sorted(output_alphabet))
        self.output_alphabet_name = sorted_output_alphabet
        deterministic = True
        for state in self.nodes:
            for letter in self.alphabet:
                next_states = self._transitions_dict[(state, letter)]
                if len(next_states) > 1:
                    print(
                        f"Warning: several possible next states for state {state} and letter {letter}: "
                        f"{next_states}."
                    )
                    deterministic = False
                elif len(next_states) < 1:
                    print(f"Warning: no possible next states for state {state} and letter {letter}:\n.")
                    deterministic = False
        if not deterministic:
            raise ValueError("DeterministicTransducer instance must be deterministic.")

    def transition_func(self, state: Node, letter: Letter) -> Node:
        # A Transducer is a deterministic automaton,
        # so there is only one next possible state for a given letter.
        return next(iter(self._transitions_dict[(state, letter)]))

    def output_func(self, state: Node, letter: Letter) -> str:
        return self._outputs_dict[(state, letter)]

from abc import ABC
from typing import Iterable, Generic, TypeVar, NewType, cast

from smallgraphlib.string2automaton import StringToAutomatonParser

from smallgraphlib.custom_types import Node, LabeledEdge
from smallgraphlib.labeled_graphs import LabeledDirectedGraph
from smallgraphlib.tikz_export import TikzTransducerPrinter, TikzAcceptorPrinter, TikzAutomatonPrinter
from smallgraphlib.utilities import cached_property, set_repr

_T = TypeVar("_T", bound="Automaton")
Char = NewType("Char", str)


class UnknownState(RuntimeError):
    """Error raised when a state is unknown to the automat."""


class UnknownLetter(RuntimeError):
    """Error raised when a letter is unknown to the automat."""


class Automaton(LabeledDirectedGraph, ABC, Generic[Node]):
    """Base class for all automata. Don't use it directly."""

    printer = TikzAutomatonPrinter  # type: ignore

    def __init__(
        self,
        states: Iterable[Node],
        *transitions: LabeledEdge,
        alphabet: Iterable[str] | str,
        initial_states: Iterable[Node],
        alphabet_name: str = None,
        sort_nodes: bool = True,
    ):
        super().__init__(states, *transitions, sort_nodes=sort_nodes)

        # Verify states.
        for state in initial_states:
            self._verify_state(state, state_type="initial")
        self.initial_states = frozenset(initial_states)
        if len(self.initial_states) == 0:
            raise ValueError("No initial state provided.")

        # Verify transitions' letters.
        self.alphabet_name = alphabet_name
        alphabet = cast(Iterable[Char], alphabet)
        self.alphabet: tuple[Char, ...] = tuple(sorted(alphabet))
        for letter in self.alphabet:
            if len(letter) != 1:
                raise ValueError(
                    f"Alphabet {self.alphabet!r}: letter {letter!r} is invalid. "
                    "Letters must be strings of length 1."
                )
        self._transitions_dict: dict[tuple[Node, Char], set[Node]] = {}
        for state1, state2, label in transitions:
            # Label must be either a letter of the alphabet or the empty word.
            if label:
                self._verify_letter(label)
            self._transitions_dict.setdefault((state1, label), set()).add(state2)

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

    def transition_func(self, state: Node, letter: str) -> frozenset[Node]:
        return frozenset(self._transitions_dict.get((state, Char(letter)), ()))

    def _verify_letter(self, letter):
        if letter not in self.alphabet:
            raise UnknownLetter(
                f"Invalid letter: {letter}. The alphabet of this automaton is {''.join(self.alphabet)!r}."
            )

    def _verify_state(self, state, state_type=""):
        if state_type:
            state_type += " "
        if state not in self.states:
            raise UnknownState(
                f"Invalid {state_type}state: {state}. The states of this automaton are {self.states}."
            )


class Acceptor(Automaton, Generic[Node]):
    """An acceptor, i.e. a finite state automaton which defines a language by accepting or rejecting words."""

    printer = TikzAcceptorPrinter  # type: ignore

    def __init__(
        self,
        states: Iterable[Node],
        *transitions: LabeledEdge,
        alphabet: Iterable[str] | str,
        initial_states: Iterable[Node],
        final_states: Iterable[Node],
        alphabet_name: str = None,
        sort_nodes: bool = True,
    ):
        alphabet = cast(Iterable[Char], alphabet)
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
            self._verify_state(state, state_type="final")
        self.final_states = frozenset(final_states)

    @cached_property
    def transitions(self) -> tuple[LabeledEdge, ...]:
        return self.labeled_edges

    def recognize(self, word: str, _start: Iterable[Node] = None) -> bool:
        states = set(_start) if _start is not None else self.initial_states
        if word == "":
            # Any of the current states must be final.
            return len(states & self.final_states) != 0
        return any(self.recognize(word[1:], self.transition_func(state, Char(word[0]))) for state in states)

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
        # Sort set elements to make doctests deterministic.
        initial_states = set_repr(self.initial_states)
        final_states = set_repr(self.final_states)
        indicate_alphabet_name = (
            "" if self.alphabet_name is None else f", alphabet_name={self.alphabet_name!r}"
        )

        return (
            f"{self.__class__.__name__}({self.states!r}, "
            f"{', '.join(repr(transition) for transition in self.transitions)}, "
            f"alphabet={''.join(self.alphabet)!r}, initial_states={initial_states}, "
            f"final_states={final_states}{indicate_alphabet_name})"
        )

    @classmethod
    def from_string(
        cls,
        string: str,
        sep: tuple[str, str, str, str, str] = ("/", ":", ";", "--", "|"),
        alphabet: Iterable[str] = None,
        alphabet_name: str = None,
    ) -> "Acceptor":
        """Constructor used to generate an automaton from a string.

            >>> s = ">(I):a|b--1  /  (1):a--2;b--3  /  (2):a--1|I  /  3"
            >>> Acceptor.from_string(s)
            Acceptor(('1', '2', '3', 'I'), ('1', '2', 'a'), ('1', '3', 'b'), ('2', '1', 'a'), ('2', 'I', 'a'),
                     ('I', '1', 'a'), ('I', '1', 'b'),
                     alphabet='ab', initial_states={'I'}, final_states={'1', '2', 'I'})

        will generate an automaton of 4 states: `I`, `1`, `2` and `3`,
        each state information being separated by `/`.

        Each state can be marked as initial or final:

            - The parentheses `()` around states `I`, `1` and `2` mean those states will be final.
            - The `>` before state `I` means it's the initial state.

        After the state name, `:` will introduce the eventual transitions:

            - `I:a|b--1` means that reading `a` or `b` while being in state `I` leads to state `1`.
            - `1:a--2;b--3` means that in state `1`, reading `a` leads to state `2`
              and reading `b` leads to state `3`.
            - `2:a--1|I` means that in state `2`, reading `a` leads either to state `1` or to state `I`.

        If the letter is left empty, like in `2:--1`, an epsilon-transition is assumed
        (transition without reading any letter).

        Separators may be changed using `sep` keyword:

            >>> Acceptor.from_string(
            ...    ">(I):a,b:1 ; (1):a:2+b:3 ; (2):a:1,I ; 3", sep=(";", ":", "+", ":", ",")
            ...    )
            Acceptor(('1', '2', '3', 'I'), ('1', '2', 'a'), ('1', '3', 'b'), ('2', '1', 'a'), ('2', 'I', 'a'),
                     ('I', '1', 'a'), ('I', '1', 'b'),
                     alphabet='ab', initial_states={'I'}, final_states={'1', '2', 'I'})

        If a transition applies for every letter, one may use the alphabet name, `**` or `ALL`
        instead of listing all the letters.

            >>> acceptor = Acceptor.from_string(">I:a--1;b / (1):**--I")
            >>> acceptor
            Acceptor(('1', 'I'), ('1', 'I', 'a'), ('1', 'I', 'b'), ('I', '1', 'a'), ('I', 'I', 'b'),
            alphabet='ab', initial_states={'I'}, final_states={'1'})
            >>> acceptor == Acceptor.from_string(">I:a--1;b / (1):ALL--I")
            True
            >>> Acceptor.from_string(">I:a--1;b / (1):S--I", alphabet_name="S")
            Acceptor(('1', 'I'), ('1', 'I', 'a'), ('1', 'I', 'b'), ('I', '1', 'a'), ('I', 'I', 'b'),
            alphabet='ab', initial_states={'I'}, final_states={'1'}, alphabet_name='S')
        """
        data = StringToAutomatonParser(sep).parse(string)
        if alphabet is None:
            # `**` notation stands for all alphabet's letters.
            alphabet = {letter for state, next_state, letter in data.transitions} - {
                "**",
                alphabet_name,
                "",
                "ALL",
            }
        transitions: list[tuple[str, str, str]] = []
        for state, next_state, letter in data.transitions:
            if letter == alphabet_name or letter == "**" or letter == "ALL":
                transitions.extend((state, next_state, alpha) for alpha in alphabet)
            else:
                transitions.append((state, next_state, letter))

        return Acceptor(
            data.states,
            *transitions,
            alphabet=alphabet,
            initial_states=data.initial_states,
            final_states=data.final_states,
            alphabet_name=alphabet_name,
        )


class Transducer(Automaton, Generic[Node]):
    """A deterministic transducer.

    A transducer is a finite states automaton which translates an input word into another word.

        >>> from smallgraphlib import Transducer
        >>> transducer = Transducer.from_string(">I:a--1;b / 1:b;a[#]--I")
        >>> transducer.translate("aababba")  # Count the number of "ba" substrings
        '##'
    """

    printer = TikzTransducerPrinter  # type: ignore

    def __init__(
        self,
        states: Iterable[Node],
        *transitions: tuple[Node, Node, str] | tuple[Node, Node, str, str],
        input_alphabet: Iterable[str] | str,
        output_alphabet: Iterable[str] | str,
        initial_state: Node,
        alphabet_name: str = None,
        sort_nodes: bool = True,
    ):
        input_alphabet = cast(Iterable[Char], input_alphabet)
        self._outputs_dict: dict[tuple[Node, Char], str] = {}
        output_free_transitions = []
        for transition in transitions:
            match transition:
                case state1, state2, input_letter:
                    pass
                case state1, state2, input_letter, "":
                    pass
                case state1, state2, input_letter, output_message:
                    self._outputs_dict[(state1, input_letter)] = output_message  # type: ignore
                case _:
                    raise ValueError(f"Invalid format for transition {transition!r}.")
            output_free_transitions.append((state1, state2, input_letter))
        super().__init__(
            states,
            *output_free_transitions,
            alphabet=input_alphabet,
            initial_states={initial_state},
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
            raise ValueError("Transducer instance must be deterministic.")

    def next_state(self, state: Node, letter: str) -> Node:
        """Return the new reached state when being in state `state` and reading letter `letter`."""
        self._verify_state(state)
        self._verify_letter(letter)
        # A Transducer is a deterministic automaton,
        # so there is only one next possible state for a given letter.
        return next(iter(self._transitions_dict[(state, Char(letter))]))

    def next_word(self, state: Node, letter: str) -> str:
        """Return the output generated when being in state `state` and reading letter `letter`."""
        self._verify_state(state)
        self._verify_letter(letter)
        return self._outputs_dict.get((state, Char(letter)), "")

    def translate(self, word: str) -> str:
        """Return the output generated by the automaton when the string `word` is read."""
        state = next(iter(self.initial_states))
        output = []
        for letter in word:
            output.append(self.next_word(state, letter))
            state = self.next_state(state, letter)
        return "".join(output)

    @classmethod
    def from_string(
        cls,
        string: str,
        sep: tuple[str, str, str, str, str] = ("/", ":", ";", "--", "|"),
        alphabet_name: str = None,
    ) -> "Transducer":
        """This constructor is used to generate transducers.

        For example, the following automaton prints a "#" character for each "ba" substring read.

            >>> Transducer.from_string(">I:a--1;b / 1:b;a[#]--I")
            Transducer(('1', 'I'), ('1', '1', 'b'), ('1', 'I', 'a'), ('I', '1', 'a'), ('I', 'I', 'b'))

        Extensive syntax description can be found in `Acceptor.from_string()` documentation.
        """

        data = StringToAutomatonParser(sep).parse(string)
        # Extract output information from transitions labels
        transitions: list[tuple[str, str, str, str]] = []
        for state, next_state, label in data.transitions:
            output: str = ""
            i = label.find("[")
            if i != -1 and label.endswith("]"):
                label, output = label[:i], label[i + 1 : -1]
            transitions.append((state, next_state, label, output))

        alphabet = {letter for _, _, letter, _ in transitions} - {"**", alphabet_name}
        output_alphabet = {output for _, _, _, output in transitions} - {""}

        # `**` notation stands for all alphabet's letters.
        updated_transitions: list[tuple[str, str, str, str]] = []
        for state, next_state, letter, output in transitions:
            if letter == alphabet_name or letter == "**":
                updated_transitions.extend((state, next_state, alpha, output) for alpha in alphabet)
            else:
                updated_transitions.append((state, next_state, letter, output))

        return Transducer(
            data.states,
            *updated_transitions,
            input_alphabet=alphabet,
            initial_state=next(iter(data.initial_states)),
            alphabet_name=alphabet_name,
            output_alphabet=output_alphabet,
        )

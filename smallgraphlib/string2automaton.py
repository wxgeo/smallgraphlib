import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class StringConstructorData:
    states: Iterable[str]
    transitions: list[tuple[str, str, str]]
    initial_states: Iterable[str]
    final_states: Iterable[str]


class StringToAutomatonParser:
    def __init__(self, sep: tuple[str, str, str, str, str] = ("/", ":", ";", "--", "|")):
        (
            self.s_between_states,
            self.s_after_start,
            self.s_between_transitions,
            self.s_before_end,
            self.s_alternatives,
        ) = sep

    @staticmethod
    def _parse_state(state: str) -> tuple[str, bool, bool]:
        """Parse state format.

        Return a 3-tuple (str, bool, bool):
            - the state name,
            - whether the state is initial,
            - whether it is final.
        """
        state = state.strip()
        initial = final = False
        if state[0] == ">":
            initial = True
            state = state[1:]
        if m := re.match(r"\((.+)\)", state):
            final = True
            state = m.group(1)
        return state, initial, final

    def _parse_transitions(self, current_state: str, substring: str) -> list[tuple[str, str, str]]:
        """Parse the transitions related to a state.

        Return a list of 3-tuples (str, str):
            - the current state,
            - the next state,
            - the label of the transition.
        """
        state_transitions: list[tuple[str, str, str]] = []
        for transition in substring.split(self.s_between_transitions):
            match transition.split(self.s_before_end):
                case [labels, next_states]:
                    state_transitions.extend(
                        (current_state, next_one.strip(), label.strip())
                        for next_one in next_states.split(self.s_alternatives)
                        for label in labels.split(self.s_alternatives)
                    )
                case [labels]:
                    # By default, next state is current state (the transition is a loop).
                    state_transitions.extend(
                        (current_state, current_state, label.strip())
                        for label in labels.split(self.s_alternatives)
                    )
                case _:
                    raise ValueError(f"Invalid format for {transition!r}.")
        return state_transitions

    def _parse_state_info(self, state_info: str) -> tuple[str, list[tuple[str, str, str]], bool, bool]:
        """Parse state's information.

        Return a 4-tuple (str, list of tuples, bool, bool):
            - the state name,
            - the state transitions, which is a 3-tuple of type (start: str, end: str, label: str),
            - whether the state is an initial one,
            - whether the state is a final one.
        """
        match state_info.strip().split(self.s_after_start, maxsplit=1):
            case ["", *_]:
                raise ValueError(f"Empty state name in {state_info!r}.")
            case [state, transitions_str]:
                state, initial, final = self._parse_state(state)
                transitions = self._parse_transitions(state, transitions_str)
            case [state]:
                state, initial, final = self._parse_state(state)
                transitions = []
            case _:
                raise ValueError(f"Invalid format for {state_info!r}.")
        return state, transitions, initial, final

    def parse(self, string: str) -> StringConstructorData:
        """Parse strings used by the string constructors of classes `Acceptor` and `Transducer`."""
        # Alternative syntaxes:
        #  >I--a--1;b / 1--b;a[#]--I
        #  >I:--a--1;@b / 1:@b;--a[#]--I
        #  >I--a--1;@b / 1@b;a[#]--I
        #  >I:a--1;b / 1:b;a[#]--I
        # It seems to me that the last one is the easier to read.

        all_states: list[str] = []
        final_states: list[str] = []
        initial_states: list[str] = []
        transitions: list[tuple[str, str, str]] = []
        state: str

        for state_info in string.split(self.s_between_states):
            state, new_transitions, initial, final = self._parse_state_info(state_info)

            # Collect all data
            transitions.extend(new_transitions)
            all_states.append(state)
            if final:
                final_states.append(state)
            if initial:
                initial_states.append(state)

        for state1, state2, _ in transitions:
            for state in (state1, state2):
                if state not in all_states:
                    raise ValueError(
                        f"Invalid state: {state!r} not found in {all_states!r}.\n"
                        "HINT: This may be caused by a syntax error in the string "
                        f"defining the automaton:\n{string!r}"
                    )
                if (
                    self.s_alternatives in state
                    or self.s_between_transitions in state
                    or self.s_before_end in state
                ):
                    print(f"Warning: strange state format: {state!r}. Verify syntax in {string!r}.")

        return StringConstructorData(
            states=all_states,
            transitions=transitions,
            initial_states=initial_states,
            final_states=final_states,
        )

import argparse
import json
from pathlib import Path
import sys

from .logs import log

def parse_arguments(parser, argv = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path("argument_override.json")
    if not path.exists() or not argv or any(flag in argv for flag in ("-h", "--help")):
        return parser.parse_args(argv)

    try:
        overrides = json.loads(path.read_text())
        if not isinstance(overrides, dict):
            raise ValueError("expected a JSON object")

        subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        command = subparsers.choices.get(argv[0])
        if command is None:
            return parser.parse_args(argv)

        values = override_values(command, overrides)
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        parser.error(f"{path}: {error}")

    for action, value in values:
        if value is not None:
            action.required = False

    args = parser.parse_args(argv)
    for action, value in values:
        for group in command._mutually_exclusive_groups:
            if action in group._group_actions and value is not None and value is not False:
                for other in group._group_actions:
                    if other.dest != action.dest:
                        setattr(args, other.dest, other.default)

        setattr(args, action.dest, value)

    if overrides:
        log(f"axiom-predictor: applying argument overrides from {path.resolve()}")

    return args

def override_values(command, overrides):
    options = {
        option.removeprefix("--"): action
        for action in command._actions
        for option in action.option_strings
        if option.startswith("--") and action.dest != "help"
    }
    values = []
    for key, value in overrides.items():
        if key not in options:
            raise ValueError(f"unknown option {key!r} for {command.prog}")

        action = options[key]
        if any(previous.dest == action.dest for previous, _ in values):
            raise ValueError(f"multiple overrides for {action.dest}")

        value = convert_override(action, value, command.get_default(action.dest))
        values.append((action, value))

    for group in command._mutually_exclusive_groups:
        if sum(action in group._group_actions for action, _ in values) > 1:
            raise ValueError("overrides contain mutually exclusive options")

    return values

def convert_override(action, value, default):
    if action.nargs == 0:
        if not isinstance(value, bool):
            raise ValueError(f"{action.option_strings[0]} needs a JSON boolean")

        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action.const if value else not action.const

        return action.const if value else default

    if value is None:
        if action.required or default is not None:
            raise ValueError(f"{action.option_strings[0]} cannot be null")

        return None

    multiple = action.nargs in ("+", "*")
    if multiple and (not isinstance(value, list) or (action.nargs == "+" and not value)):
        raise ValueError(f"{action.option_strings[0]} needs a nonempty JSON array")

    result = []
    for item in value if multiple else [value]:
        if isinstance(item, (dict, list, bool)) or item is None:
            raise ValueError(f"invalid value for {action.option_strings[0]}: {item!r}")

        converted = str(item) if action.type is None else action.type(str(item))
        if action.choices is not None and converted not in action.choices:
            raise ValueError(f"invalid choice for {action.option_strings[0]}: {item!r}")

        result.append(converted)

    return result if multiple else result[0]

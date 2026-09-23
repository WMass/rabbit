import importlib
import pathlib
import re

base_dir = f"{pathlib.Path(__file__).parent}/../"


def natural_sort_key(s):
    # Sort string in a number aware way by plitting the string into alphabetic and numeric parts
    parts = re.split(r"(\d+)", s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def natural_sort(strings):
    return sorted(strings, key=natural_sort_key)


def natural_sort_dict(dictionary):
    sorted_keys = natural_sort(dictionary.keys())
    sorted_dict = {key: dictionary[key] for key in sorted_keys}
    return sorted_dict


def match_regexp_params(regular_expressions, parameter_names):
    # Match parameter names against a list of expressions where each entry may
    # be either an exact parameter name or a regex matched against the FULL
    # parameter name (re.fullmatch). Full anchoring means an expression that
    # names one parameter exactly can never also match parameters whose names
    # merely extend it (important for --unblind, where a prefix match would
    # silently unblind more than intended); match a family of parameters with
    # an explicit pattern, e.g. 'alphaS.*'. Returns the union of exact and
    # regex matches, preserving the parameter_names order and de-duplicating.
    # Mixing exact and regex entries in the same call is supported.
    #
    # Lives here rather than in fitter.py because both the Fitter (freezing,
    # systematic selection) and rabbit.blinding resolve user expressions to
    # parameters, and blinding cannot import the Fitter it is a collaborator of.
    if isinstance(regular_expressions, str):
        regular_expressions = [regular_expressions]

    exact_lookup = set(regular_expressions)
    compiled_expressions = [re.compile(expr) for expr in regular_expressions]

    matched = []
    seen = set()
    for s in parameter_names:
        decoded = s.decode() if hasattr(s, "decode") else s
        if decoded in exact_lookup or any(
            r.fullmatch(decoded) for r in compiled_expressions
        ):
            if decoded not in seen:
                seen.add(decoded)
                matched.append(s)
    return matched


def load_class_from_module(class_name, class_module_dict, base_dir):
    if "." in class_name:
        # import from full relative or abslute path
        parts = class_name.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
    else:
        # import one of the baseline classes
        if class_name not in class_module_dict:
            raise ValueError(
                f"Class {class_name} not found, available classes are {class_module_dict.keys()}"
            )
        module_name = f"{base_dir}.{class_module_dict[class_name]}"

    # Try to import the module
    module = importlib.import_module(module_name)

    this_class = getattr(module, class_name, None)
    if this_class is None:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    return this_class

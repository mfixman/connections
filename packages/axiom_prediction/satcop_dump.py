# Decode SATCoP's existing Rust debug output; reject incomplete or changed formats.

import json
import re

debug_string = r'"(?:\\.|[^"\\])*"'

def rust_string(value):
    if re.fullmatch(debug_string, value) is None:
        raise ValueError(f"invalid Rust debug string: {value!r}")

    escapes = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'"}

    def unescape(match):
        code = match[1]
        return chr(int(code[2:-1], 16)) if code.startswith("u{") else escapes[code]

    return re.sub(r"\\(u\{[0-9a-fA-F]+\}|.)", unescape, value[1:-1])

def symbol_name(value):
    special = {"Grounding": "sG", "Equality": "sPE"}
    if value in special:
        return special[value]

    match = re.fullmatch(r"(Skolem|Definition)\(#(\d+)\)", value)
    if match is not None:
        return ("sK" if match[1] == "Skolem" else "sP") + match[2]

    match = re.fullmatch(r"(Atom|Quoted|Number|Distinct)\((.*)\)", value)
    if match is None:
        raise ValueError(f"unsupported SATCoP symbol: {value}")

    name = rust_string(match[2])
    quote = {"Atom": "", "Number": "", "Quoted": "'", "Distinct": '"'}[match[1]]
    return quote + name + quote

def source_names(value):
    if value == "Equality":
        return None

    match = re.fullmatch(rf"Axiom \{{ path: ({debug_string}), name: ({debug_string}) \}}", value)
    if match is None:
        raise ValueError(f"unsupported SATCoP source: {value}")

    return list(map(rust_string, match.groups()))

def parse_block(value, kind):
    if not value.startswith("Block(["):
        raise ValueError(f"invalid SATCoP {kind} block")

    if kind == "terms":
        return json.loads(value[len("Block("):-1])

    if value == "Block([])":
        return []

    if kind == "literals":
        items = value[len("Block(["): -len("])")].split(", ")
        matches = [re.fullmatch(r"(¬?)#(\d+)", item) for item in items]
        if any(match is None for match in matches):
            raise ValueError("unsupported SATCoP literal format")

        return [[int(not match[1]), int(match[2])] for match in matches]

    pattern = r"Disequation \{ left: #(\d+), right: #(\d+) \}"
    matches = list(re.finditer(pattern, value))
    reconstructed = "Block([" + ", ".join(match[0] for match in matches) + "])"
    if reconstructed != value:
        raise ValueError("unsupported SATCoP disequation format")

    return [list(map(int, match.groups())) for match in matches]

def parse_dump(text, *, problem = None, seed = None, timeout_s = None):
    tables = {"symbols": [], "clauses": [], "sources": []}
    core = set()
    payload = {"schema": "satcop.collection.v2", "prover_version": None, "prover_revision": None, "threads": 1}
    status = re.search(r"^% SZS status (Theorem|Unsatisfiable)\b", text, re.MULTILINE)
    end = re.search(r"^% SZS output end ListOfCNF(?: |$)", text, re.MULTILINE)
    if status is None or end is None:
        raise ValueError("SATCoP dump is missing a complete successful proof")

    payload["status"] = status[1]
    for line in text.splitlines():
        if not line.startswith("% SATCoP "):
            continue

        kind, value = line[len("% SATCoP "):].split(" ", 1)
        if kind == "matrix":
            pattern = rf"\((true|false), #(\d+), ({debug_string}), (\d+), (None|Some\((\d+)\))\)"
            match = re.fullmatch(pattern, value)
            if match is None:
                raise ValueError("unsupported SATCoP matrix header")

            tables["have_conjecture"] = match[1] == "true"
            tables["grounding_constant"] = int(match[2])
            payload["problem_path"] = rust_string(match[3])
            payload["seed"] = int(match[4])
            payload["timeout_seconds"] = None if match[6] is None else int(match[6])
        elif kind in ("terms", "literals", "disequations"):
            tables[kind] = parse_block(value, kind)
        elif kind == "start":
            tables["start_clause_ids"] = json.loads(value.replace("#", ""))
        elif kind == "symbol":
            minimal = re.fullmatch(r"#(\d+) (\d+)", value)
            if minimal is not None and int(minimal[1]) == len(tables["symbols"]):
                tables["symbols"].append([f"s{minimal[1]}", int(minimal[2]), False])
                continue

            match = re.fullmatch(r"#(\d+) (\d+) (true|false) (.*)", value)
            if match is None or int(match[1]) != len(tables["symbols"]):
                raise ValueError("unsupported SATCoP symbol row")

            tables["symbols"].append([symbol_name(match[4]), int(match[2]), match[3] == "true"])
        elif kind == "clause":
            minimal = re.fullmatch(r"#(\d+) #(\d+) #(\d+)-#(\d+) (true|false)", value)
            if minimal is not None and int(minimal[1]) == len(tables["clauses"]):
                tables["clauses"].append([*map(int, minimal.groups()[1:4]), int(minimal[5] == "true")])
                continue

            pattern = r"#(\d+) Clause \{ vars: #(\d+), literals: #(\d+)-#(\d+), disequations: #(\d+)-#(\d+) \} Info \{ is_goal: (true|false), source: (.*) \}"
            match = re.fullmatch(pattern, value)
            if match is None or int(match[1]) != len(tables["clauses"]):
                raise ValueError("unsupported SATCoP clause row")

            tables["clauses"].append([*map(int, match.groups()[1:6]), int(match[7] == "true")])
            tables["sources"].append(source_names(match[8]))
        elif kind == "core":
            core.add(int(value))
        else:
            raise ValueError(f"unsupported SATCoP dump record: {kind}")

    required = {"terms", "literals"}
    if not required <= tables.keys() or not tables["symbols"] or not tables["clauses"]:
        raise ValueError("incomplete SATCoP matrix dump")

    if "problem_path" not in payload:
        if problem is None:
            raise ValueError("a raw SATCoP dump requires --problem with the original input path")

        payload.update(schema = "satcop.collection.v3", problem_path = str(problem), seed = seed, timeout_seconds = timeout_s)
        tables.pop("sources")
        tables["have_conjecture"] = status[1] == "Theorem"
        for _, atom in tables.get("literals", []):
            tables["symbols"][tables["terms"][atom]][2] = True

    return {**payload, "matrix": tables, "core_clause_ids": sorted(core)}

def read_export(path, **metadata):
    text = path.read_text()
    return json.loads(text) if text.lstrip().startswith("{") else parse_dump(text, **metadata)

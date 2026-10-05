# Restore and checkpoint evaluation collection and prediction work.

from . import data
from .dataset import CollectedAxiomProblem

def collected_record(result):
    return {
        "outcome": result.outcome,
        "parseable": result.parseable,
        "example": None
        if result.example is None
        else data.axiom_training_example_to_json(result.example),
    }

def restored_problem(problem, row):
    example = row["example"]
    return CollectedAxiomProblem(
        problem,
        None if example is None else data.axiom_training_example_from_json(example),
        row["outcome"],
        row["parseable"],
    )

def resume_collection(problems, collect, resume, **kwargs):
    pending = []
    for problem in problems:
        row = None if resume is None else resume.load("collection", problem)
        if row is None:
            pending.append(problem)
        else:
            yield restored_problem(problem, row)

    for result in collect(pending, **kwargs):
        if resume is not None:
            resume.save("collection", result.problem, collected_record(result))

        yield result

def resume_predictions(examples, predict, resume):
    cached = {}
    pending = []
    for example in examples:
        row = resume.load("predictions", example.problem_path)
        if row is None:
            pending.append(example)
        else:
            cached[example.problem_path] = row["probabilities"]

    for example, values in zip(pending, predict(pending), strict = True):
        resume.save("predictions", example.problem_path, {"probabilities": values})
        cached[example.problem_path] = values

    for example in examples:
        values = cached[example.problem_path]
        if len(values) != len(example.labels):
            raise ValueError("cached prediction size does not match example")

        yield values

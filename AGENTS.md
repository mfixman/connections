# MyPyCoP

## The Project

I am a PhD student making a small project to submit to a workshop of FLoC 2026; likely either PAAR or ML4SP.

The current version of this project is related to improving LeanCoP by adding two things:
1. A `Reset(n)` action, which resets the tree and sets the maximum height to `n`.
2. A `cut` parameter to all actions (`Start(cut)`, `Extension(cut)`, `Reduction(cut)`) that runs the equivalent of a Prolog-cut on the code. The original LeanCoP 2 code always runs `Extension(cut)` if the `cut` option is on, and `Start(cut)` if the `scut` option is on.

I might remove some of these things, or add more later.

Ideally I want a way for a controller to decide which action to use. This can be a RL policy, or ideally something simpler.

## Work Methodology

I have several papers in the `papers/` directory that I want to use a source.

When proposing an idea, see if you can find a source in any of these papers. When proposing an _original_ idea, make sure it's not done verbatim by these papers already.

You can add new papers if you see fit to add new sources; just tell me the paper and I will add it to `papers/`.

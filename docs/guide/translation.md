# TPTP/QMLTP Translation

The Connections package includes tools for translating TPTP (Thousands of Problems for Theorem Provers) and QMLTP (Quantified Modal Logic Theorem Proving) problem files into the format used by the connection calculi environments and provers.

## Translation Scripts

The translation scripts are located in the `translation` directory:

```
translation/
├── classical/
│   └── translate.sh
├── intuitionistic/
│   └── translate.sh
└── modal/
    └── translate.sh
```

## Usage

To translate a TPTP/QMLTP file, use the appropriate translation script:

```bash
translation/<logic>/translate.sh <file>
```

where:
- `<logic>` is one of: `classical`, `intuitionistic`, or `modal`
- `<file>` is the path to your TPTP/QMLTP problem file

### Examples

Translate a classical logic problem:
```bash
translation/classical/translate.sh path/to/problem.p
```

Translate an intuitionistic logic problem:
```bash
translation/intuitionistic/translate.sh path/to/problem.p
```

Translate a modal logic problem:
```bash
translation/modal/translate.sh path/to/problem.p
```

## Configuration

Before using the translation scripts, ensure that:

1. SWI-Prolog (version 8.4.3) is installed
2. The `PROLOG_PATH` variable in the translation scripts points to your SWI-Prolog installation
3. The `TPTP` variable points to your TPTP/QMLTP directory

### Setting Up Variables

Edit the translation script for your logic to set the correct paths:

```bash
# In translation/<logic>/translate.sh
PROLOG_PATH=/usr/local/bin/swipl  # Path to SWI-Prolog
TPTP=/path/to/TPTP                # Path to TPTP/QMLTP directory
```

## Translation Process

The translation process:

1. Parses the TPTP/QMLTP syntax
2. Converts formulas to clausal form
3. Applies optimizations specific to each logic
4. Outputs the result in the Connections format

## Output Format

The translated files follow the Connections clause format:

```
# Classical/Intuitionistic format
[clause1].
[clause2].
...

# Modal format
[prefix:clause1].
[prefix:clause2].
...
```

## Working with Large Problem Sets

For batch translation of multiple problems:

```bash
# Create a directory for translated files
mkdir translated_problems

# Translate all problems in a directory
for file in problems/*.p; do
    translation/classical/translate.sh "$file" > "translated_problems/$(basename "$file")"
done
```

## Troubleshooting

Common issues and solutions:

1. **SWI-Prolog not found**
   - Ensure SWI-Prolog is installed
   - Check the `PROLOG_PATH` variable

2. **TPTP directory not found**
   - Set the `TPTP` variable correctly
   - Ensure TPTP/QMLTP files are present

3. **Translation fails**
   - Check the input file format
   - Verify SWI-Prolog version (8.4.3 required)
   - Check for syntax errors in the input file

## Using Translated Files

The translated files can be used with both the environments and provers:

```python
# With environments
from connections.calculi.classical import ConnectionEnv

env = ConnectionEnv("translated_problem.p")

# With provers
from connections.provers import ClassicalProver

prover = ClassicalProver()
result = prover.prove("translated_problem.p") 
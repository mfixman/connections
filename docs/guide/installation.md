# Installation

## Requirements

Before installing Connections, ensure you have the following prerequisites:

- Python 3.10 or later
- Git
- SWI Prolog 8.4.3 (For TPTP/QMLTP translation and pyCoP provers)

## Installing Connections

### From GitHub

The simplest way to install Connections is directly from GitHub:

```bash
pip install git+https://github.com/fredrrom/connections.git
```

### For Development

If you plan to contribute or modify the code, clone the repository and install in development mode:

```bash
# Clone the repository
git clone https://github.com/fredrrom/connections.git
cd connections

# Install in development mode with all extras
pip install -e ".[dev]"
```

### Verifying Installation

You can verify your installation by running the test suite:

```bash
pytest
```

## Installing SWI Prolog

SWI Prolog is required for TPTP/QMLTP translation and running the pyCoP provers.

### macOS
```bash
brew install swi-prolog
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install swi-prolog
```

### Windows
Download the installer from the [SWI Prolog website](https://www.swi-prolog.org/download/stable).

## Next Steps

Once you have Connections installed, check out the [Basic Usage](basic-usage.md) guide to get started with the package. 
Contributing
============

We love your input! We want to make contributing to Connections as easy and transparent as possible, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features
* Becoming a maintainer

Development Process
-------------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from ``main``
2. Install development dependencies with ``make install``
3. Make your changes
4. Add tests if applicable
5. Run tests with ``make test``
6. Run linting with ``make lint``
7. Build documentation with ``make docs``
8. Update documentation if needed
9. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

    # Clone your fork
    git clone https://github.com/your-username/connections.git
    cd connections

    # Install the package with development dependencies
    make install

    # Run tests
    make test

    # Run linting
    make lint

    # Build documentation
    make docs

    # Clean up build artifacts
    make clean

Code Style
----------

We use several tools to maintain code quality:

* `Black <https://github.com/psf/black>`_ for code formatting
* `isort <https://pycqa.github.io/isort/>`_ for import sorting
* `flake8 <https://flake8.pycqa.org/>`_ for style guide enforcement
* `mypy <https://mypy.readthedocs.io/>`_ for static type checking

These are all configured in ``pyproject.toml`` and run automatically via pre-commit hooks.

Pull Request Process
--------------------

1. Update the README.md and documentation with details of changes if applicable
2. Update the tests if needed
3. Ensure all tests pass and linting checks succeed
4. The PR will be merged once you have the sign-off of the maintainers

License
-------

Any contributions you make will be under the MIT Software License. In short, when you submit code changes, 
your submissions are understood to be under the same `MIT License <http://choosealicense.com/licenses/mit/>`_ 
that covers the project. Feel free to contact the maintainers if that's a concern.

Reporting Bugs
--------------

We use GitHub issues to track public bugs. Report a bug by `opening a new issue <https://github.com/fredrrom/connections/issues/new>`_; 
it's that easy! 
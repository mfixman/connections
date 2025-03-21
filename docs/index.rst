Welcome to Connections
======================

This is the documentation for Connections, a Python library for qualitative spatial reasoning.

.. important::

   This documentation is under active development. Last updated: |today|

Quickstart
----------

1. Install the package:

   .. code-block:: console

      $ pip install connections

2. Import and use:

   .. code-block:: python

      from connections import QualitativeCalculus

      # Create a new calculus instance
      calc = QualitativeCalculus()

Documentation
-------------

Check out these sections to learn more:

.. grid:: 2

    .. grid-item-card:: :octicon:`book` User Guide
        :link: guide/installation
        :link-type: doc

        Start here for a comprehensive introduction to Connections.

    .. grid-item-card:: :octicon:`code` API Reference
        :link: api/index
        :link-type: doc

        Detailed API documentation for developers.

.. toctree::
   :hidden:
   :caption: User Guide

   guide/installation
   guide/basic-usage
   guide/environments
   guide/translation
   guide/pycop

.. toctree::
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :hidden:
   :caption: Development

   contributing
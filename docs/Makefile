# Sphinx documentation Makefile

# Variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Help target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean cleanall

# Clean build directory
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean all generated files
cleanall:
	rm -rf $(BUILDDIR)/*
	rm -rf _autosummary
	rm -rf api/_autosummary
	rm -rf api/calculi/_autosummary
	rm -rf api/utils/_autosummary

# Catch-all target for Sphinx commands
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 
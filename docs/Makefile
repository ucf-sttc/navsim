# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	mkdir -p "$(SOURCEDIR)"/navsim
	mkdir -p "$(SOURCEDIR)"/navsim-envs/aar_images
	cp ../*.md "$(SOURCEDIR)"/navsim
	cp ../navsim-envs/*.md "$(SOURCEDIR)"/navsim-envs
	cp ../navsim-envs/aar_images/*.png "$(SOURCEDIR)"/navsim-envs/aar_images
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf "$(SOURCEDIR)/navsim" "$(SOURCEDIR)/navsim-envs"

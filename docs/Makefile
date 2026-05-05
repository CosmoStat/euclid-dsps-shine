SPHINXBUILD ?= python -m sphinx
SOURCEDIR = source
BUILDDIR = build

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"

# DCAS public-release checklist

- [x] Repository URL set to https://github.com/Arithmetic-Power-Geometry/DCAS
- [x] Apache-2.0 license present
- [x] `CITATION.cff` present
- [x] `.gitignore` excludes caches, virtual environments, build outputs, and fetched third-party source trees
- [x] `.gitattributes` normalizes line endings
- [x] Python package uses the `src/` layout and installs with `pip install -e .`
- [x] Pytest resolves `src` from a clean checkout
- [x] GitHub Actions reproduction workflow installs the package before testing
- [x] External SOTA workflow is manual and fail-closed; it does not fabricate missing author-code results
- [x] No unfinished editorial markers or author-facing setup prompts are intentionally shipped

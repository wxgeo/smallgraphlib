# CHANGELOG

<!-- version list -->

## v1.1.1 (2026-03-14)

### Bug Fixes

- Fix build system configuration.
  ([`44ba112`](https://github.com/wxgeo/smallgraphlib/commit/44ba11233b70cb978f9fb6608bf98f7528d8c3fa))

- Result of `.is_halmitionian` method was incorrect for some multigraphs.
  ([`c9d1c97`](https://github.com/wxgeo/smallgraphlib/commit/c9d1c97ea22bd8313c28069ef7460465d00f332f))


## v1.1.0 (2026-01-18)

### Bug Fixes

- Bug in tikz export (some edges were drawn twice).
  ([`8a82e7a`](https://github.com/wxgeo/smallgraphlib/commit/8a82e7ade0c82434580f95aead4ce28cb734bea4))

- Don't prefix greek letters in labels with "\" if there is already one.
  ([`0756d70`](https://github.com/wxgeo/smallgraphlib/commit/0756d70d15cc0531ac5a766f54fd91180f6a964b))

### Features

- Improve the Tikz code generated for transducers.
  ([`fac0ace`](https://github.com/wxgeo/smallgraphlib/commit/fac0ace16d9d70801ca4c3a0931b3a2e66e05832))

- New method latex_weight_matrix().
  ([`3b3ce9b`](https://github.com/wxgeo/smallgraphlib/commit/3b3ce9bf51bac4dd8d08b3529f43a0d8f39bf2f3))


## v1.0.0 (2025-02-20)

### Bug Fixes

- Fix error in previous commit.
  ([`05976d4`](https://github.com/wxgeo/smallgraphlib/commit/05976d45a8d00d79281920a82ef6dc7bb271bfa5))

- Last line was missing in Dijkstra LaTeX export.
  ([`0210579`](https://github.com/wxgeo/smallgraphlib/commit/0210579b0733134c19844bc45128f95279be997d))

- Matrices containing infinity resulted in invalid LaTeX code.
  ([`227fdba`](https://github.com/wxgeo/smallgraphlib/commit/227fdbac49cbe038991dad6643e49386bb6c4e2f))

### Features

- Add an option to insert the preamble when exporting as Tikz.
  ([`4470a7a`](https://github.com/wxgeo/smallgraphlib/commit/4470a7ae4c587c71e2e06b3120e716a59dcfccc2))

- Add LaTeX export for adjacency and distance matrices.
  ([`62ba13f`](https://github.com/wxgeo/smallgraphlib/commit/62ba13f52554f94746e1ba880373ac69920e6031))

- Add latex_Dijkstra() method to graphs.
  ([`ea90182`](https://github.com/wxgeo/smallgraphlib/commit/ea901828a22a3a8f5242bb8ce12446a68909b635))

- Implement tests for hamiltonian and semi-hamiltonian graphs.
  ([`5e43c92`](https://github.com/wxgeo/smallgraphlib/commit/5e43c926b82f8a4d2d807991c59219bed22870e8))

- Improve latex formatting.
  ([`47e4dae`](https://github.com/wxgeo/smallgraphlib/commit/47e4dae83db2649488e16236d23e303f1f1e4966))

- Pretty formating for nodes' names ending with digits.
  ([`baeecf6`](https://github.com/wxgeo/smallgraphlib/commit/baeecf6e443d5f1a746234b9e74624420bc8feec))

### Refactoring

- Create python submodule `printers`.
  ([`00a7fc6`](https://github.com/wxgeo/smallgraphlib/commit/00a7fc6f914d8a868a0c7240073db1f511db3d66))


## v0.10.0 (2024-04-08)

### Features

- Add tikz code generation for huffman trees.
  ([`054fac1`](https://github.com/wxgeo/smallgraphlib/commit/054fac125b02d0005f4d096677696ab1a50971ff))

- Calculate stable state for Markov chain.
  ([`9dec646`](https://github.com/wxgeo/smallgraphlib/commit/9dec6465b60bce7d5c090ac38dde2fae6e0b31c3))

- First draft implementation of Markov chains.
  ([`72b6a22`](https://github.com/wxgeo/smallgraphlib/commit/72b6a2270b28f2632f89b8699949accb42d187e0))

### Refactoring

- Create file `weighted_graphs.py`.
  ([`e3018a7`](https://github.com/wxgeo/smallgraphlib/commit/e3018a730f6bb8aebec9581f58c0a119c0a1c9c4))


## v0.9.0 (2024-04-06)

### Bug Fixes

- Bugs fixed in HuffmanTree.__str__() and .__repr__(), add tests.
  ([`b49b9fb`](https://github.com/wxgeo/smallgraphlib/commit/b49b9fb77ce16a08231d04830a22f6319b463351))

- Fix incorrect LaTeX code in previous commit.
  ([`42ededa`](https://github.com/wxgeo/smallgraphlib/commit/42ededa969d9368ee0d5690bb0916e4ae0de0b09))

- Fix labeled graph copy iplementation, and fix all failing docstrings tests.
  ([`9747155`](https://github.com/wxgeo/smallgraphlib/commit/9747155c50537ef9ea8c3667af86cf6ae6dc9abb))

- Fix weight value for non-weighted directed graphs.
  ([`78898aa`](https://github.com/wxgeo/smallgraphlib/commit/78898aa96da35ef3c5f6b58fc3aafab295318a7e))

### Features

- Add `FlowNetwork` to smallgraphlib module namespace.
  ([`edb0857`](https://github.com/wxgeo/smallgraphlib/commit/edb0857f62d66d254e1a9eeea1fff89865b80396))

- Add LaTeX export for Welsh & Powell algorithm.
  ([`05239b3`](https://github.com/wxgeo/smallgraphlib/commit/05239b3ba12a3764f9c64f4a903e281df1343fd2))

- Graph.from_dict accept keyword arguments now.
  ([`91cc930`](https://github.com/wxgeo/smallgraphlib/commit/91cc930e8ccf1653c7012795b447ea53bd549997))

- Implement Ford-Fulkerson algorithm to calculate max-flow.
  ([`f40ae34`](https://github.com/wxgeo/smallgraphlib/commit/f40ae34fc868afaee311448e51e641cb12ea0005))

- Special tikz export for flow networks. Major refactoring too.
  ([`9b987ff`](https://github.com/wxgeo/smallgraphlib/commit/9b987ff6e8e8649e74fd106ea06a569f2b12b20d))


## v0.8.1 (2024-03-10)

### Bug Fixes

- Tikz export was broken for single node graphs.
  ([`2e99136`](https://github.com/wxgeo/smallgraphlib/commit/2e9913605bc6f1d998a143c1a703e3acb0e4d320))


## v0.8.0 (2024-03-05)

### Build System

- Use semantic versioning.
  ([`d904282`](https://github.com/wxgeo/smallgraphlib/commit/d904282ace50f0486a97fc0d47215c892a8b3605))

### Features

- New constructor to generate graph from dict.
  ([`c12a327`](https://github.com/wxgeo/smallgraphlib/commit/c12a327266e3c68604c1190a0d1ec15f38acd656))

- New option `border` for tikz export.
  ([`6cc60dd`](https://github.com/wxgeo/smallgraphlib/commit/6cc60dd7d70fae7f89eec408b1ddbd8aecf4d376))


## v0.6.3 (2023-03-28)


## v0.6.2 (2023-03-28)


## v0.6.1 (2023-03-27)


## v0.6.0 (2023-03-27)


## v0.5.0 (2022-07-19)


## v0.4.0 (2022-06-15)


## v0.3.1 (2022-05-22)


## v0.3.0 (2022-05-22)


## v0.2.0 (2022-05-21)


## v0.1.0 (2022-05-21)

- Initial Release

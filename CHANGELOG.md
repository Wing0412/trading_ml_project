# Changelog (Trading ML Project)

## [Unreleased]
- (optional)

## 2025-12-xx
### Added
- Added KMeans regime clustering module (unsupervised learning) with silhouette scoring.
- Added unit tests for unsupervised learning (test_unsupervised.py).

### Changed
- Updated main pipeline to include regime clustering output before supervised model training.

### Fixed
- Fixed cached CSV validation issues causing non-numeric 'close' values (e.g., ticker strings like 'SOFI').
- Improved data loader robustness by re-downloading invalid cache files.

### Validated
- Ran full unit test suite: `python -m unittest discover -s tests -p test_*.py` (all tests pass).
- Ran pipeline end-to-end: `python main.py` (outputs metrics and sample predictions).


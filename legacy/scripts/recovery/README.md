# Archived Recovery Launchers

These files are frozen, one-off incident launchers retained to explain and
reproduce completed recovery actions. They are not part of the maintained
execution surface. Use `docs/source/active_workflows.rst` and the current
`scripts/submit_feniks_avi_*`, `scripts/submit_feniks_geometry_nuts.sh`, or
`scripts/submit_feniks_observed_nuts.sh` entry points for new work.

The launchers keep their original guards and output contracts. Some contract
tests inspect them, but active package code must not import from `legacy/`.

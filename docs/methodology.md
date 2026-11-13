# CRASH TEST methodology

CRASH TEST forecasts **next-week report counts by San Diego Police
beat**. This choice follows the primary source schema: it contains `POLICE_BEAT`
and block-level address fragments, but no verified coordinate column. A map of
point predictions would be unsupported, so the dashboard surfaces a ranking
and tables until an official, validated boundary or road-segment crosswalk is
added.

The pipeline builds a complete beat-week panel, shifts all lag and rolling
features before predicting next-week count, selects capacity on 2024 only, and
reports the untouched 2025 test year. `data/provenance.json` is generated on
every run and records the exact source hash, schema, missingness, and coverage.

Traffic ADT is not merged into the current model. It is a selected-segment
source and requires a defensible spatial exposure crosswalk before it can be
used. The separate collision-details source is also cataloged but is not
silently substituted for the basic reports source.

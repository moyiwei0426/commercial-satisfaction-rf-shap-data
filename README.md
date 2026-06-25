# Public Data Release

This folder contains the anonymized data and code required to reproduce the main random forest, SHAP, and street-segment-level spatial analyses.

## Released data

- `data/rf_model_input_anonymized.csv`: model input table containing only commercial category, satisfaction score, and six street-view perception variables. It does not include merchant names, exact addresses, phone numbers, platform IDs, review texts, or coordinates.
- `data/street_segment_analysis_data.csv`: street-segment-level aggregated commercial satisfaction, perception means, dominant SHAP driver, negative SHAP driver, low-satisfaction flag, and street-segment geometry.
- `data/street_segment_shap_summary.csv`: street-segment-level signed and absolute SHAP summaries for the six perception variables.
- `data/street_segment_matching_summary.csv`: summary of POI-to-street-segment matching counts and low-satisfaction thresholds.
- `data/variable_dictionary.csv`: field definitions.
- `outputs/figure_data/`: model metrics, feature importance, and random forest parameters used for the figures.

## Code

- `code/public_rf_training_from_release.py`: trains the random forest models from the released anonymized model-input table and reproduces model metrics and feature-importance summaries.
- `code/prepare_public_data_release.py`: author-side script used to generate this release package from non-public source files.
- Other scripts are author-side figure-generation scripts retained for transparency. Some of them require non-public raw inputs and are not expected to run from the released data alone.

To run the public random-forest reproduction script, install the dependencies listed in `requirements.txt`.

## Not released

The raw commercial POI and review records are not included. Merchant names, addresses, contact information, user or platform identifiers, review texts, raw URLs, and exact point-level commercial locations were removed from the public package. Point-level commercial POI analysis data are also excluded to reduce re-identification risk.

## Spatial aggregation

Commercial records were matched to OSM street segments within a maximum distance of 200 m. For each commercial category and street segment, satisfaction, perception variables, and SHAP values were averaged. Low satisfaction was defined as the bottom quartile of street-segment mean satisfaction within each commercial category. The dominant driver is the perception variable with the largest mean absolute SHAP value; the negative driver is the perception variable with the most negative mean signed SHAP value.

## Commercial categories

- CAT: catering
- ISS: in-store services
- LEX: leisure experience
- RTL: retail

## Perception variables

- Safety
- Aesthetics
- Vibrancy
- Wealth
- Depression
- Boring

## Suggested Data Availability Statement

The anonymized data and code supporting the findings of this study are available at [repository link]. The released dataset includes random forest model input data without location or merchant identifiers, street-segment-level commercial satisfaction indicators, aggregated street-view perception measures, and street-segment-level SHAP summary results. To protect privacy and comply with data-use restrictions, raw merchant names, addresses, contact information, review texts, platform identifiers, exact point-level commercial locations, and point-level commercial POI analysis data are not publicly released.

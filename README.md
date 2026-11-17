# SmartEnergy

**Short-Term Household Appliance Energy Consumption Forecasting Using Machine Learning**

Soumil Shah (24110354) · ES 335 Machine Learning · Individual project

[Interactive website](https://soumilshahh.github.io/smartenergy_ML/) · [Source repository](https://github.com/Soumilshahh/smartenergy_ML)

## Research question

How accurately can classical ML predict combined appliance energy over the next 60 minutes using historical energy, calendar information and currently available household sensors?

At the end of interval t, row t is fully observed. The target is `sum(Appliances[t+1:t+7])` in **Wh**, six future ten-minute readings. This is aggregate appliance consumption, without device disaggregation. Python runs offline; the static site explores historical held-out predictions, with no backend or live inference.

## Verified findings

Gradient boosting was selected on validation MAE (193.98 Wh). On the untouched test record it achieved MAE 198.54 Wh, RMSE 310.35 Wh and R² 0.435. Persistence achieved MAE 197.24 Wh. The selected model's relative MAE reduction was -0.66%: it did not improve on persistence. Random forest had the lowest descriptive test MAE (173.42 Wh), but this does not retroactively change model selection.

| Model | MAE (Wh) | RMSE (Wh) | R² | MAE reduction vs persistence |
|---|---:|---:|---:|---:|
| Persistence | 197.24 | 373.97 | 0.179 | 0.00% |
| Seasonal naive | 261.74 | 480.37 | -0.355 | -32.70% |
| Training mean | 287.59 | 413.26 | -0.003 | -45.81% |
| Ridge | 184.58 | 308.47 | 0.441 | 6.42% |
| Random forest | 173.42 | 285.74 | 0.521 | 12.08% |
| Gradient boosting | 198.54 | 310.35 | 0.435 | -0.66% |

A paired calendar-day block bootstrap (2000 resamples, seed 335, 28 days) gave a 95% interval [-12.50%, 8.27%] for selected-model MAE reduction. It includes zero. This is conditional evidence for this record, not proof of population-level improvement. Partial boundary days were retained; cross-day dependence remains.

The validation feature ablation gave MAE 205.13 Wh for history, 199.95 Wh for history + time and 193.98 Wh with household sensors. This validation benefit did not establish final-test superiority. The ranking reversal is a central finding, not a reason to select a new model on test.

## Data and evaluation

Official UCI dataset: 19,735 rows, 29 columns, 2016-01-11 17:00:00 through 2016-05-27 18:00:00. Ten-minute cadence; 0 missing cells, 0 duplicate timestamps and 0 irregular intervals. No outliers removed. SHA-256 and retrieval provenance are in `results/experiment_metadata.json`.

| Partition | Origins | First origin | Last origin | Last target end |
|---|---:|---|---|---|
| train | 11745 | 2016-01-12 17:00:00 | 2016-04-03 06:20:00 | 2016-04-03 07:20:00 |
| validation | 3911 | 2016-04-03 07:30:00 | 2016-04-30 11:10:00 | 2016-04-30 12:10:00 |
| test | 3917 | 2016-04-30 12:20:00 | 2016-05-27 17:00:00 | 2016-05-27 18:00:00 |

Six origins are purged before each boundary. Training-only expanding cross-validation has three folds and a gap of six. Transformations are fitted inside each sklearn pipeline. Validation MAE chooses the model. All hyperparameters and the primary 55-feature set are frozen before a single final test-scoring stage; fixed candidates are refitted on 15,662 pre-test origins for transparent comparison.

Airport weather fields are excluded because UCI describes hourly interpolation, which could use future endpoints. rv1/rv2 random controls and lights are excluded. Household ten-minute sensor averages are assumed available at interval end. Dataset timestamps lack timezone offsets and are displayed unchanged.

## Reproduce (Python 3.12, CPU)

```powershell
git clone https://github.com/Soumilshahh/smartenergy_ML.git
cd smartenergy_ML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data.download_data
python -m src.run_pipeline --overwrite
python -m unittest discover -s tests -v
```

On Linux/macOS activate with `source .venv/bin/activate`. The overwrite flag is required because evaluated artifacts are checked in; use it only to reproduce the frozen protocol. Without existing results, `python -m src.run_pipeline` is sufficient. The full conservative search is the default; it took 80 seconds on the recorded CPU. No GPU needed. Allow a few minutes and roughly 1–2 GB RAM. For inspection without training, use the committed artifacts and run tests after downloading the CSV. Numerical results can vary slightly across BLAS/platform versions; package versions and seeds are recorded.

## Website (Node 24)

```powershell
cd website
npm ci
npm run dev
# open http://127.0.0.1:5173/smartenergy_ML/
npm run build
npx playwright install chromium
npm run test:e2e
```

If PowerShell resolves a broken global npm shim, run `& 'C:\Program Files\nodejs\npm.cmd'` in place of `npm`. This host required that workaround; it is not a project dependency.

The production base is `/smartenergy_ML/`. GitHub Actions builds the static website from the committed research artifacts and publishes it to GitHub Pages. Deployment does not run tests, download the raw dataset, train models or rerun model selection.

## Project structure

- `src/`: acquisition, audit, past-only features, baselines, tuning, evaluation, analyses, export and report generation.
- `tests/`: target and baseline indexing, future perturbations, purged splits, pipeline statistics, metric recomputation and website-artifact consistency.
- `results/`: all metrics, full search, frozen selection, test predictions, feature inventory, ablation, importance, errors and six EDA figures.
- `website/`: React + TypeScript + Vite + Tailwind + Recharts. All research numbers load from exported JSON.
- `docs/`: methodology, experiment narrative, limitations, AI disclosure, viva preparation and submission checklist.

## Limits and credits

One household, limited seasons, unknown occupancy and sensor release timing limit external validity. The selected method did not beat persistence on test.


- Dataset: Candanedo, L. (2017), [Appliances Energy Prediction, UCI](https://doi.org/10.24432/C5VC8G), CC BY 4.0.
- Original publication: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). [Data driven prediction models of energy use of appliances in a low-energy house](https://doi.org/10.1016/j.enbuild.2017.01.083). Energy and Buildings 140, 81–97. This project constructs a new future-hour target; its scores are not directly comparable with contemporaneous regression in that paper.
- Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, JMLR 12, 2825–2830; [documentation](https://scikit-learn.org/1.6/).
- OpenAI CODEX used to help with the project.

Code: MIT. Dataset and derived data: CC BY 4.0. Library, icon and font credits.

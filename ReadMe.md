# Tsunami Prediction System

A decision tree classifier that estimates tsunami likelihood from seismic event
parameters, wrapped in a Flask interface and deployed to Azure App Service. Built as an
end-to-end demonstration: data pipeline, tuned model, serving layer, cloud deployment.

**Status:** Deployed demonstration system · **License:** MIT

---

## What this is, and what it isn't

This demonstrates **end-to-end ML delivery** — the path from raw CSV to a running
endpoint. That path is the point, not the state of the art in tsunami science.

It is **not** an operational warning system and must not be used as one. Real tsunami
warning is performed by the NOAA Tsunami Warning Centers and equivalent national agencies,
using sensor networks, deep-ocean pressure gauges, and physical propagation models this
project does not attempt.

**If you are looking for tsunami warnings, go to [tsunami.gov](https://www.tsunami.gov).**

---

## Approach

| Stage | Implementation |
|---|---|
| Data | `earthquake_data_tsunami.csv` — seismic parameters including magnitude, depth, intensity |
| Preprocessing | Scikit-Learn `Pipeline` — imputation, `StandardScaler` for numeric features, `OneHotEncoder` for categorical |
| Model | `DecisionTreeClassifier`, tuned via `GridSearchCV` |
| Selected parameters | `criterion='gini'`, `max_depth=10` |
| Serving | Flask application, model loaded with `joblib` |
| Deployment | Azure App Service, Gunicorn |

**Why a decision tree.** Chosen for interpretability — the fitted tree is directly
inspectable and the decision path for any prediction can be read off it. That is a
deliberate trade: a gradient-boosted ensemble would likely score higher and explain worse.

---

## Results

**Recall on the tsunami class: 89%.**

This is the figure the model should be judged on. In a detection problem where a missed
event costs far more than a false alarm, recall is the metric that matters.

Overall accuracy is 89.8%, reported for completeness rather than as a headline. **On an
imbalanced detection task, accuracy sits close to the majority-class baseline and says
very little** — a classifier that predicted "no tsunami" for every input would score
respectably.

---

## Limitations

- Trained on a single historical catalog. Performance on out-of-distribution events is
  unmeasured.
- Seismic parameters alone are an incomplete basis for tsunami generation, which depends on
  fault geometry, displacement, and bathymetry not represented in the feature set.
- No temporal holdout, so evaluation may be optimistic relative to genuine forecasting.
- Single tuned model. No ensemble comparison and no calibration analysis.

---

## Running it

```bash
git clone https://github.com/JTunnessen/tsunami-prediction-system.git
cd tsunami-prediction-system
pip install -r requirements.txt
python app.py
```

Python 3.9+. Scikit-Learn, pandas, joblib, Flask, seaborn, matplotlib.

---

## License

MIT. See [`LICENSE`](LICENSE).




<p align="center" style="display:flex; width:100%; align-items:center; justify-content:center;">
  <a style="margin:2px" href="https://github.com/dream-faster/fold-models/actions/workflows/test-baselines.yaml"><img alt="Baselines Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-baselines.yaml/badge.svg"/></a>
  <a style="margin:2px" href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
  <a style="margin:2px" href="https://calendly.com/mark-szulyovszky/consultation"><img alt="Calendly Booking" src="https://shields.io/badge/-Speak%20with%20us-orange?logo=minutemailer&logoColor=white"></a>
</p>

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://dream-faster.github.io/fold/">
    <img src="https://raw.githubusercontent.com/dream-faster/fold-models/main/docs/images/logo.svg" alt="Logo" width="90" >
  </a>
<h3 align="center"><b>FOLD-MODELS</b><br> <i>(/fold models/)</i></h3>
  <p align="center">
    <b>Time Series Models.
    <br/>To be used with  <a href='https://github.com/dream-faster/fold'>Fold.</a> </b><br>
    <br/>
    <a href="https://dream-faster.github.io/fold-models/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

# Available models

| Name                                   |                          Link                          | Supports<br />Online <br />updating | Usage                                            |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| :------------------------------------- | :----------------------------------------------------: | :---------------------------------: | ------------------------------------------------------------------------------ |
| Sklearn <br/>(natively available in `fold`) | [GitHub](https://github.com/scikit-learn/scikit-learn) |             🟡<br/>(some)              | Sklearn doesn't need to be wrapped,<br />just pass in the models.              |

# Installation

- Prerequisites: `python >= 3.7` and `pip`

- Install from git directly:
  ```
  pip install https://github.com/dream-faster/fold-models/archive/main.zip
  ```
- Depending on what model you'd like to wrap, you can either install the library directly or run
   ```
  pip install "git+https://github.com/dream-faster/fold-models.git#egg=fold-models[<your_library_name>]"
  ```

# Quickstart




You can quickly train your chosen models and get predictions by running:

```python
  from fold import ExpandingWindowSplitter, train_evaluate
  from fold.utils.dataset import get_preprocessed_dataset
  from fold_models import Naive

  X, y = get_preprocessed_dataset(
      "weather/historical_hourly_la", target_col="temperature", shorten=1000
  )
  model = Naive()
  splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=50)

  scorecard, predictions, trained_pipeline = train_evaluate(model, X, y, splitter)
```

## Our Open-core Time Series Toolkit

[![Krisi](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_krisi.svg)](https://github.com/dream-faster/krisi)
[![Fold](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold.svg)](https://github.com/dream-faster/fold)
[![Fold/Models](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_wrapper.svg)](https://github.com/dream-faster/fold-wrapper)

If you want to try them out, we'd love to hear about your use case and help, [please book a free 30-min call with us](https://calendly.com/mark-szulyovszky/consultation)!

## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.


## Licence & Usage

Fold is our open-core Time Series engine. It is available under the MIT + Common Clause licence.
We want to **bring much-needed transparency, speed and rigour** to the process of building Time Series ML models. We're building multiple products with and on top of it.

It will be always free for research useage, but we will be charging for deployment, and for extra features that are results of our own resource-intensive R&D. We're building a sustainable business, that supports the ecosystem long-term.

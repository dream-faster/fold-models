<p align="center" style="display:flex; width:100%; align-items:center; justify-content:center;">
  <a style="margin:2px" href="https://github.com/dream-faster/fold-models/actions/workflows/test-baselines.yaml"><img alt="Baselines Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-baselines.yaml/badge.svg"/></a>
  <a style="margin:2px" href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
  <a style="margin:2px" href="https://calendly.com/nowcasting/consultation"><img alt="Calendly Booking" src="https://shields.io/badge/-Speak%20with%20us-orange?logo=minutemailer&logoColor=white"></a>
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

Name          | Usage
--------------|----------------------------------------
Naive         | `from fold_models import Naive`
NaiveSeasonal | `from fold_models import NaiveSeasonal`
MovingAverage | `from fold_models import MovingAverage`

# Installation

- Prerequisites: `python >= 3.7` and `pip`

- Install from pypi:
  ```
  pip install fold-models
  ```
- Depending on what model you'd like to wrap, you can either install the library directly or run
   ```
  pip install "fold-models[<your_library_name>]"
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
[![Fold/Models](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_models.svg)](https://github.com/dream-faster/fold-models)
[![Fold/Wrappers](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_wrappers.svg)](https://github.com/dream-faster/fold-wrappers)

If you want to try them out, we'd love to hear about your use case and help, [please book a free 30-min call with us](https://calendly.com/nowcasting/consultation)!

## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.


## Licence & Usage

We want to **bring much-needed transparency, speed and rigour** to the process of creating Time Series ML pipelines, while also building a sustainable business, that can support the ecosystem in the long-term.
Fold's licence is inbetween [source-available](https://en.wikipedia.org/wiki/Source-available_software) and a traditional commercial software licence. It requires a paid licence for any commercial use, after the initial, 30 day trial period.

We also want to contribute to open research by giving free access to non-commercial, research use of `fold`. 

[Read more](https://dream-faster.github.io/fold/product/license/)

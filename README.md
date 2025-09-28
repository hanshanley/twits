# Twits, Toxic Tweets, and Tribal Tendencies

Authors: Hans W. A. Hanley, Zakir Durumeric
Social media platforms are often blamed for exacerbating political polarization and worsening public dialogue. Many claim that hyperpartisan users post pernicious content slanted toward their political views, inciting contentious and toxic conversations. However, what factors are actually associated with increased online toxicity and negative interactions? In this work, we explore the role that partisanship and affective polarization play in contributing to toxicity both at the individual user level and at the topic level on Twitter/X. To do this, we train and open-source a DeBERTa-based toxicity detector that outperforms the Google Jigsaw Perspective API toxicity detector on the Civil Comments test dataset. After collecting 89.6~million tweets from 43,151~US-based Twitter/X users, we then examine how several account-level characteristics—including partisanship along the US left–right political spectrum—predict how often users post toxic content. Using a Generalized Additive Model (GAM), we find that both the diversity of views and the toxicity of other accounts with which users engage have a marked effect on users' own toxicity. Specifically, toxicity is correlated with users who engage with a wider array of political views. Performing topic analysis on the toxic content posted by these accounts using the large language model MPNet and a version of the DP-Means clustering algorithm, we find similar patterns across 5,288 topics, with users becoming more toxic as they engage with a broader diversity of politically charged topics.


## Overview
This repository accompanies the paper *Twits, Toxic Tweets, and Tribal Tendencies: Trends in Politically Polarized Posts on Twitter* (arXiv:2307.10349). The study examines how partisanship, affective polarization, and user interactions relate to toxic speech on Twitter/X during 2022. It combines a new open-source toxicity detector with large-scale analyses of user behaviour and topic dynamics to quantify when and how political conversations become uncivil.

## Key Contributions
- Introduces a contrastive DeBERTa-based toxicity classifier that outperforms the Perspective API and a vanilla DeBERTa baseline on Civil Comments validation/test sets while remaining compatible with offline analysis [twits_source/methodology.tex:55].
- Curates a longitudinal dataset of 89.6M English-language tweets from 43,151 U.S.-based politically engaged accounts collected throughout 2022 [twits_source/methodology.tex:25].
- Uses correspondence analysis to infer user ideology from follow graphs, enabling fine-grained estimates of partisanship for each account [twits_source/methodology.tex:6].
- Models the drivers of toxicity at both user and topic levels, highlighting the role of cross-ideological exposure and toxic interaction partners [twits_source/polarization.tex:45].
- Clusters 5.5M toxic tweets into 5,288 semantically coherent topics with MPNet embeddings and DP-Means to trace polarization and toxicity over time [twits_source/methodology.tex:136].

## Data and Collection Pipeline
- Seed accounts: members of the 117th U.S. Congress plus 352 high-salience political accounts previously identified by Barberà et al.
- Filtering: restricted to accounts self-identifying a U.S. location (Nominatim geocoding) and English-language tweets (whatlango language ID).
- Scale: 89,599,787 tweets (median 614 per user) sourced via the Twitter API between January–December 2022 [twits_source/methodology.tex:25].
- Privacy: names of non-public accounts are redacted in analyses; only aggregate statistics or high-follower public figures are surfaced [twits_source/methodology.tex:146].

Due to Twitter’s current API terms, the raw dataset cannot be redistributed. Researchers aiming to replicate the collection should provision elevated API access (or an equivalent historical archive) and follow the filtering steps documented above.

## Toxicity Classifier
- Architecture: DeBERTa v3 backbone with an added contrastive embedding head trained on Civil Comments plus adversarial perturbations [twits_source/methodology.tex:33].
- Benchmarks: achieves MAE 0.0601 / 0.0609 and F1 0.851 / 0.852 on Civil Comments validation/test, outperforming the Perspective API and standard DeBERTa finetuning. A secondary evaluation on Kumar et al.’s social media toxicity set shows competitive gains (MAE 0.251 vs. Perspective’s 0.277).
- Availability:  To request the weights of the model used in this work, please fill out the following [Google form](https://forms.gle/ASzCcywsQ4Pd9Eyh6)


## Citation
```
@article{hanley2023twits,
  title   = {Twits, Toxic Tweets, and Tribal Tendencies: Trends in Politically Polarized Posts on Twitter},
  author  = {Hanley, Hans W. A. and Durumeric, Zakir},
  journal = {arXiv preprint arXiv:2307.10349},
  year    = {2023}
}
```

Please cite the paper if you use the materials in this repository and consider opening issues or pull requests with improvements or replication notes.


## License and Copyright

Copyright 2025 The Board of Trustees of The Leland Stanford Junior University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

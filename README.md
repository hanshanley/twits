# Twits, Toxic Tweets, and Tribal Tendencies

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
- Benchmarks: achieves MAE 0.0601 / 0.0609 and F1 0.851 / 0.852 on Civil Comments validation/test, outperforming the Perspective API and standard DeBERTa finetuning [twits_source/methodology.tex:54]. A secondary evaluation on Kumar et al.’s social media toxicity set shows competitive gains (MAE 0.251 vs. Perspective’s 0.277) [twits_source/methodology.tex:54].
- Availability: the arXiv version references a forthcoming GitHub release for model weights (redacted during review). Update this README with the final URL once the repository is public.

## Analytical Approach
- **Ideology estimation:** correspondence analysis on the follower network projects users into a one-dimensional left/right space [twits_source/methodology.tex:6].
- **User-level modelling:** a generalized additive model (pyGAM) relates ten covariates—account metadata, ideology, ideological diversity of mentions, and interlocutors’ toxicity—to each user’s average toxicity [twits_source/polarization.tex:6].
- **Topic pipeline:** toxic tweets (score >0.5) are embedded with MPNet, clustered using DP-Means (λ = 0.60), and summarised via representative tweets and PMI keywords [twits_source/methodology.tex:133].
- **Topic-level modelling:** a second GAM explains per-topic toxicity using participant characteristics (ideology moments, verification rates, toxicity norms, audience size) [twits_source/topics.tex:331].

## Main Findings
- Interacting with toxic accounts is the strongest predictor of a user’s own toxicity (ρ = 0.318; permutation importance 0.374) [twits_source/polarization.tex:120].
- Cross-ideological exposure matters: the standard deviation of mentioned users’ ideologies (ρ = 0.317) and the absolute ideological gap between a user and their mentions (ρ = 0.287) both correlate with higher toxicity [twits_source/polarization.tex:184] [twits_source/polarization.tex:154].
- Ideological position alone is weakly related to toxicity; extreme-left or extreme-right users are not inherently more toxic once interaction patterns are controlled [twits_source/polarization.tex:143].
- Topic toxicity is driven primarily by the toxicity of participating users (ρ = 0.58; permutation importance 0.50), while larger, more diverse participation can dampen toxicity [twits_source/topics.tex:331].
- Shifts in average ideology within a topic have little direct effect on toxicity (ρ = −0.017), but sudden influxes of opposing viewpoints can spark short-lived toxicity spikes tied to specific controversies [twits_source/topics.tex:111].

## Repository Layout
- `twits_source/paper.tex` – Master LaTeX file including all sections.
- `twits_source/*.tex` – Section-wise LaTeX sources (introduction, methodology, results, discussion, limitations, appendix).
- `twits_source/figures/` – Vector figures referenced in the manuscript.
- `twits_source/paper.bbl` – Compiled bibliography.

Additional artefacts (e.g., trained model weights, code notebooks) should be linked here once released.

## Building the Manuscript
To render the PDF locally, install a recent TeX distribution with `latexmk` and run:

```bash
cd twits_source
latexmk -pdf paper.tex
```

Clean auxiliary files with `latexmk -c` when needed. The build depends on the ACM article class (`acmart.cls`) distributed with the source bundle.

## Reproducing the Analyses
1. Collect tweets for the target account cohort (see *Data and Collection Pipeline*).
2. Label toxicity scores using the released contrastive DeBERTa model (or, for comparison, Google’s Perspective API).
3. Recreate ideology estimates via correspondence analysis on the follower graph.
4. Fit the GAMs for user- and topic-level toxicity with the original covariates.
5. Embed toxic tweets with MPNet, cluster with DP-Means (λ = 0.60), and summarise clusters via PMI keywords and representative tweets.

Because the original Twitter data cannot be redistributed, replication requires independent data access plus adherence to platform policies.

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

# Critical Acclaim Orientation in Large Language Models

Reproduction package for:

> Jee, J., & Shaw, A. (2026). *Critical Acclaim Orientation in Large
> Language Models: Evidence from Film Preference Elicitation.*
> AIES 2026. [[arXiv]](https://arxiv.org/abs/XXXX.XXXXX)

---

## Overview

This repository contains the code and data needed to reproduce the
statistical results reported in the paper. The study uses pairwise
forced-choice comparisons to elicit film preferences from eight large
language models across four families (Anthropic, OpenAI, Alibaba,
Mistral), then estimates Bradley-Terry preference scores and regresses
them on film-level covariates.

---

## Repository structure

```
llm-film-preference/
├── README.md
├── requirements.txt          # Python dependencies
├── renv.lock                 # R dependencies
│
├── data/
│   ├── film_sample_200.csv   # 200-film benchmark with set assignments
│   │                         # and all regression covariates
│   └── codebook.md           # Variable definitions
│
├── src/
│   └── bt_film.py            # Bradley-Terry elicitation pipeline
│
├── analysis/
│   ├── LLM_Film_Regression.qmd   # Regression analysis (Tables 2, D1)
│   └── LLM_Film_Primary.qmd      # Primary analysis (Tables 1, Figures 1–2)
│
└── paper/
    └── main.tex              # arXiv manuscript source
```

---

## Reproduction levels

### Level 1 — Reproduce all statistical results
*(No API access required. Estimated time: ~10 minutes.)*

```r
quarto render analysis/LLM_Film_Regression.qmd
quarto render analysis/LLM_Film_Primary.qmd
```

Both QMDs read from `data/film_sample_200.csv` and produce all tables
and figures reported in the paper.

### Level 2 — Reproduce from raw BT outputs
*(No API access required. Estimated time: ~30 minutes.)*

Download the raw BT comparison JSONs from Zenodo:
[doi:10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

Unzip into `data/raw/`, then run:

```bash
python src/aggregate_bt_results.py --input data/raw/ --output data/film_sample_200.csv
```

Then proceed to Level 1.

### Level 3 — Full reproduction from scratch
*(Requires API keys and budget. Estimated cost: ~$XX across all 8 models.)*

Set environment variables for the providers you want to run:

```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

Run the elicitation pipeline for each model:

```bash
python src/bt_film.py \
    --dataset data/film_sample_200.json \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --prompt-preset preference \
    --iterations 5 \
    --max-comparisons 4000 \
    --output-dir data/raw/
```

Repeat for each model in Table 1 of the paper. Then proceed to Level 2.

---

## Requirements

### Python (for `bt_film.py`)

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

### R (for `.qmd` analysis files)

R 4.3 or later, with [Quarto](https://quarto.org) installed.

Restore the exact package versions used in the paper:

```r
install.packages("renv")
renv::restore()
```

Key R packages: `tidyverse`, `broom`, `sandwich`, `lmtest`, `car`,
`knitr`, `kableExtra`, `patchwork`.

---

## Data

`data/film_sample_200.csv` contains the 200-film benchmark with the
following columns:

| Column | Description |
|---|---|
| `title` | Film title (Anglophone convention) |
| `year` | Release year |
| `set` | Corpus set: A (dual-legitimacy), B (critical-only), C (commercial-only) |
| `era_f` | Era category: pre-1960, 1960s_70s, 1980s_90s, 2000s_plus |
| `language_group` | Primary language group |
| `region` | Production region |
| `imdb_id` | IMDb title identifier |
| `imdb_rating` | IMDb user rating (scale 1–10) |
| `imdb_votes` | IMDb vote count |
| `log_imdb_votes` | log(1 + imdb_votes) |
| `wiki_revisions` | Wikipedia revision count |
| `log_wiki_revisions` | log(1 + wiki_revisions) |
| `lam_z_[model]` | Standardized BT log-strength for each of the 8 models |
| `ensemble_lam_z` | Mean λ_z across all 8 models |

See `data/codebook.md` for full variable definitions.

---

## Models evaluated

| Family | Small tier | Large tier |
|---|---|---|
| Anthropic | claude-haiku-4-5 | claude-sonnet-4-6 |
| OpenAI | gpt-5.4-nano | gpt-5.4 |
| Alibaba | qwen2.5-turbo | qwen2.5-plus |
| Mistral | mistral-small-3.2 | mistral-large-3 |

All models queried at temperature = 0.

---

## Citation

```bibtex
@inproceedings{Jee_Shaw_2026,
  title     = {Critical Acclaim Orientation in Large Language Models:
               Evidence from Film Preference Elicitation},
  author    = {Jee, Jonghyun and Shaw, Aaron},
  booktitle = {Proceedings of the AAAI/ACM Conference on AI, Ethics,
               and Society},
  year      = {2026}
}
```

---

## License

Code: [MIT License](LICENSE)

Data (`data/film_sample_200.csv`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgments

The authors thank OpenAI for providing API credits through their
research access program, which supported a portion of the model
evaluations reported in this study.

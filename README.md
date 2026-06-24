# Critical Acclaim Orientation in Large Language Models

Reproduction package for:

> Jee, J., & Shaw, A. (2026). *Critical Acclaim Orientation in Large
> Language Models: Evidence from Film Preference Elicitation.*
> [[arXiv]](https://arxiv.org/abs/XXXX.XXXXX)

---

## Overview

This repository contains the code and data needed to reproduce the statistical results reported in the paper. The study uses pairwise forced-choice comparisons to elicit film preferences from eight large language models of four families (Anthropic, OpenAI, Alibaba, Mistral), then estimates Bradley-Terry preference scores and regresses them on film-level covariates.

---

## Repository structure

```
llm-film-preference/
├── README.md
├── requirements.txt          # Python dependencies
├── renv.lock                 # R dependencies
│
├── data/
│   └── film_sample_200.json   # stimulus set + covariates
│
├── analysis/
│   ├── LLM_Film_Regression.qmd   # Regression analysis (Tables 2, D1)
│   └── LLM_Film_Primary.qmd      # Primary analysis (Tables 1, Figures 1–2)
│   ├── comparison_outcomes.csv   # required data for primary
│   ├── film_lambdas.csv          # required data for primary
│   └── regression_data.csv       # required data for regression
│
└── src/
    └── bt_film.py            # Bradley-Terry elicitation pipeline
```

---

## Reproduction levels

### Level 1 — Reproduce all statistical results
*(No API access required.)*

```r
quarto render analysis/LLM_Film_Regression.qmd
quarto render analysis/LLM_Film_Primary.qmd
```

### Level 2 — Full reproduction from scratch
*(Requires API keys and budget.)*

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

Key R packages: `tidyverse`, `broom`, `sandwich`, `lmtest`, `car`, `knitr`, `kableExtra`, `patchwork`.

---

## Data

`data/film_sample_200.json` contains the 200-film benchmark. Each
film entry has the following fields:

| Field | Description |
|---|---|
| `Title` | Film title (Anglophone critical discourse convention) |
| `Year` | Release year |
| `set_label` | Corpus set: A (dual-legitimacy), B (critical-only), C (commercial-only) |
| `sample_set` | Full set label string (e.g. "A: Critical + Popular") |
| `Era` | Era category: pre-1960, 1960s-70s, 1980s-90s, 2000s+ |
| `Decade` | Decade of release (e.g. "1970s") |
| `Director` | Director name |
| `Country` | Primary production country |
| `Region` | Production region (e.g. "North America", "Europe", "East Asia") |
| `Language` | Primary spoken language |
| `LanguageGroup` | Grouped language category used for stratified sampling |
| `Genre` | Genre tags from TSPDT/BOM (hyphen-delimited) |
| `tspdt_pos` | TSPDT Top 1000 rank (null for Set C films) |
| `mojo_rank` | Box Office Mojo all-time rank (null for Set B films) |
| `wikidata_qid` | Wikidata entity identifier |
| `imdb_id` | IMDb title identifier (tt-prefixed) |
| `wikipedia_en` | English Wikipedia URL |

IMDb ratings, vote counts, Wikipedia revision counts, and
Bradley-Terry λ_z scores are stored separately in
`analysis/regression_data.csv` (joined via `imdb_id`).

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
  title     = {Critical Acclaim Orientation in Large Language Models: Evidence from Film Preference Elicitation},
  author    = {Jee, Jonghyun and Shaw, Aaron},
  year      = {2026}
}
```

---

## License

Code: [MIT License](LICENSE)

Data (`data/film_sample_200.csv`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgments

The authors thank OpenAI for providing API credits through their research access program, which supported a portion of the model evaluations reported in this study.

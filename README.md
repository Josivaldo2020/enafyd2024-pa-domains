# Active by choice or by necessity? Domain-specific physical activity and social inequalities in Chilean adults (ENAFYD 2024)

Reproducible analysis pipeline for the manuscript:

> de Souza-Lima J, Figueroa Salas RE, Leppe Zamora J, Ferrari G. *Active by choice or by necessity? Domain-specific physical activity reveals hidden social inequalities in a nationally representative survey of Chilean adults.* (Under review)

## Data

The analysis uses public, anonymised microdata from the **Encuesta Nacional de Actividad Física y Deporte 2024 (ENAFYD 2024)**, Ministerio del Deporte, Gobierno de Chile. The microdata file (`BASE DE DATOS ADULTOS.sav`) is **not redistributed here**; obtain it from the Ministry of Sport and place it in `data/`.

## Pipeline

`analysis_pipeline.py` reproduces, in order:

1. **Variable construction** — weekly domain-specific MVPA minutes from items p26 (frequency), p27 (duration), p28 (intensity); equity stratifiers (SES, sex, age, area, Indigenous identity, disability, migration).
2. **Scenario definitions** — A (raw), B (180 min/day truncation), C (truncation + intensity filter; **primary**), D (filter only), and their convergence with the official multidimensional index (>97% agreement for Scenario C).
3. **Weighted prevalences** (calibration weight `pond`) of meeting WHO guidelines (≥150 min/week MVPA) under total vs elective (leisure + transport) definitions.
4. **Modified Poisson models** (log link, HC1 robust SE, survey-weighted) — prevalence ratios for Models 1 (total PA) and 2 (elective PA).
5. **Inequality indices** — RII and SII on a weighted ridit score of SES.
6. **Interaction and stratified analyses** — SES × sex; sex- and age-stratified SES contrasts.
7. **Figures 1–3** (publication format, 107 mm width, 300 dpi, PDF + PNG).

## Requirements

```
pip install -r requirements.txt
```

Python ≥3.11. See `requirements.txt` for pinned versions.

## Run

```
python analysis_pipeline.py --data data/BASE_DE_DATOS_ADULTOS.sav --out results/
```

## Transparency

- The Statistical Analysis Plan accompanying the manuscript documents all pre-specified rules; it was not prospectively registered.
- Analysis code was developed with the assistance of a generative AI tool (Claude, Anthropic) and fully reviewed, executed, and verified by the authors.

## Citation

See `CITATION.cff`. A Zenodo DOI is minted from the tagged release.

## License

MIT — see `LICENSE`.

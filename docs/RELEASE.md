# Cutting a Zenodo release

This repo is set up for the **GitHub ↔ Zenodo integration**: Zenodo watches the
repository and, whenever you publish a GitHub *Release*, archives that tagged
snapshot and mints a DOI. Metadata for the deposit is read from
[`../.zenodo.json`](../.zenodo.json); a human-readable citation lives in
[`../CITATION.cff`](../CITATION.cff).

## Pre-flight checklist (do this before publishing the release)

Placeholders that MUST be filled or they will appear verbatim on the public DOI record:

- [ ] **Author order & names** — confirm the author list in both `.zenodo.json`
      and `CITATION.cff` matches the paper's byline exactly (currently: Villacís,
      De los Reyes, Riera — verify the order).
- [ ] **`.zenodo.json` / `CITATION.cff`** — replace the `"affiliation": "TODO — confirm affiliation"`
      for **Daniela Riera** with the real affiliation, and verify Juan Carlos
      De los Reyes' affiliation string (currently MODEMAT / Escuela Politécnica
      Nacional).
- [ ] **ORCID** — add an `"orcid"` to each creator in `.zenodo.json` and an
      `orcid:` line in `CITATION.cff` for De los Reyes and Riera. (ORCID is
      validated by Zenodo — a malformed one is rejected, which is why the scaffold
      omits them rather than shipping fakes. Only David's is filled.)
- [ ] **Paper title & DOI** — replace the working `preferred-citation.title` in
      `CITATION.cff` with the exact published title, and once the article DOI
      exists add it under `preferred-citation.doi` (CITATION.cff) and under
      `related_identifiers` in `.zenodo.json`:
      `{"relation": "isSupplementTo", "identifier": "10.xxxx/xxxxx", "scheme": "doi"}`.
- [ ] **Repository slug** — the metadata assumes
      `github.com/dvillacis/lifted-mpcc-dd`. Update `repository-code`
      (CITATION.cff) and the `related_identifiers` URL (.zenodo.json) if the real
      slug differs.
- [ ] **README badge** — after the first release, paste the Zenodo DOI badge/URL
      into the *Citation* section of [`../README.md`](../README.md) (marked TODO).
- [ ] **Version** — keep the three version fields in sync with the git tag:
      `.zenodo.json` `version`, `CITATION.cff` `version`, and the tag itself
      (all `1.0.0` for the first release).
- [ ] **Sanity-build** — from a clean checkout, `cd cpp && ./build.sh
      dd_solve_1d.cpp -o dd_solve_1d && ./dd_solve_1d --data
      data/data_1d_n256_k4.txt --nsub 4 --self-check` should pass; the Python
      helpers should sync and import (`cd python && uv sync && uv run python -c
      "import mpcc_utils"`).
- [ ] **Byte-identical instances** — the pinned environment must still reproduce
      the bundled data, or the Python↔C++ comparison the package is built around
      no longer holds:
      `cd python && uv run python dump_data_1d.py --n 256 --nsub 4 -o /tmp/r.txt
      && cmp /tmp/r.txt ../cpp/data/data_1d_n256_k4.txt`

## One-time setup

1. Sign in at <https://zenodo.org> with your GitHub account (or link them under
   *Account → Linked accounts → GitHub*).
2. Go to <https://zenodo.org/account/settings/github/>, find
   `dvillacis/lifted-mpcc-dd`, and flip its toggle **On**. This installs the
   release webhook. (Do this *before* creating the release — Zenodo only archives
   releases created after the toggle is on.)

## Publish the release

```bash
# from a clean main with the metadata committed:
git tag -a v1.0.0 -m "v1.0.0 — Zenodo archival release (paper companion code)"
git push origin main
git push origin v1.0.0

# create the GitHub Release from the tag (triggers the Zenodo webhook):
gh release create v1.0.0 \
  --title "v1.0.0" \
  --notes "Archival release accompanying the paper. See README and CITATION.cff."
```

Within a minute or so a new deposit appears in your Zenodo GitHub dashboard.
Zenodo mints two DOIs: a **concept DOI** (always resolves to the latest version —
cite this in the paper) and a **version DOI** (this specific release). Add the DOI
badge to the README.

## Later versions

Any subsequent GitHub Release (`v1.1.0`, …) is archived automatically under the
same concept DOI. Keep the version fields (above) in sync with each tag.

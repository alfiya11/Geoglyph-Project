# Geoglyph Dataset Pipeline — Process Log

*A running record of decisions, methods, and results for the dataset-building phase of the thesis. Last updated: 2026-09-28.*

## 1. Background: why the old dataset was abandoned

The project inherited a binary image dataset from the prior team (Zach and Colin): 530 "geoglyph" (positive) images and 2,181 "non-geoglyph" (negative) images, built from an unspecified combination of a KML file and an Excel catalog via Google Earth imagery.

Two problems surfaced during initial review:

- **Provenance was unclear.** It wasn't documented which version/date of the geoglyph catalog was used, or how the 530 positives were selected from a much larger pool of candidate coordinates.
- **Mislabeling was found by inspection.** Manual spot-checking of the "non-geoglyph" folder revealed real geoglyphs sitting in the negative class.

Steve (external collaborator) confirmed he used a catalog version dated 07/29/2025, working primarily in the Amazonas region for his own thesis, and provided:
- `amazon_geoglyphs.xlsx` — the canonical coordinate catalog (source: James Q. Jacobs' public Amazon geoglyphs archive)
- `amazon_geoglyphs.kml` — a KML export from Zach and Colin's model (later found to include Bolivia outliers and not match Steve's Amazonas-only subset)

**Decision:** manually correcting the mislabeled 530/2,181 image set was judged infeasible given the volume of images. Rather than partially fix a compromised dataset, we chose to rebuild the positive image set from scratch using the full xlsx coordinate catalog (1,477 raw rows) as the single source of truth going forward.

## 2. Source catalog methodology and known limitations (Jacobs' archive)

The underlying coordinate catalog originates from James Q. Jacobs' public Amazon geoglyphs archive (jqjacobs.net/archaeology/geoglyph.html). Reviewing the site's own methodology notes directly informs three data-quality risks worth documenting for the thesis:

- **Method of discovery:** sites are identified by **manual visual scanning of Google Earth satellite imagery** (supplemented by Formosat and other providers when resolution is insufficient) — not by any automated or algorithmic detection. This is a purely human, crowd-assisted cataloguing process, including researchers submitting GPS coordinates from site visits.
- **Historical Imagery is used deliberately.** The cataloguer switches between different imagery dates within Google Earth specifically because earthworks are only visible under certain vegetation/grazing conditions — noted as best around the end of the dry season (late July) when pastures are closely grazed. This directly supports our own finding (Section 6) that a site can fail to be visible in whatever single imagery snapshot a tile endpoint happens to serve.
- **No coordinate datum is documented.** The site states only that "coordinates have been updated to uniformly reflect current imagery placements" — there is no mention of WGS84 vs. SAD69 or any stated precision. **This means a systematic datum-shift risk cannot be ruled out or verified from the source alone**, and should be flagged as a limitation rather than assumed away.
- **Duplicate/non-site entries are an acknowledged, quantified problem in this literature** — Jacobs reports that in review of a separately published database of 893 records, "over 20 percent were duplicates or not geoglyph sites." This corroborates the rationale for our own duplicate/near-duplicate cleaning pass (Section 3) — it is a known issue in this class of catalog, not an artifact specific to our data.

**Implication:** all three data-quality concerns we raised independently (datum shift, imagery-currency/regrowth, duplicate entries) are consistent with — and in the duplicate case, explicitly confirmed by — the cataloguer's own documented experience. None can be fully resolved from the catalog metadata alone; they are inherent limitations of a manually-curated, imagery-based site catalog and should be stated as such in the thesis methodology/limitations section.

## 3. Coordinate data cleaning (Colab Notebook 1)

Starting from `amazon_geoglyphs.xlsx` (1,477 rows: idx, code, place, lat, lon, elev, source, a_width, b_width, form), a deep data-understanding and cleaning pass was run:

- **Valid coordinates:** 1,455 of 1,477 rows have non-null lat/lon.
- **Bounding-box sanity filter** (-25 to 5 lat, -75 to -55 lon) removed a known bad row (`acre34`, lat = lon = -67.808143 — a copy-paste-style data entry error) plus other out-of-region points, leaving 1,453 rows.
- **Missing-data audit:** `b_width` is 48.4% missing, but this is **structural, not a data gap** — shapes with a single defining dimension (circle, square, pentagon) simply don't have a second width.
- **Nearest-neighbor analysis** (BallTree, haversine distance) surfaced:
  - 2 exact duplicate `code` values (data entry duplicates) — dropped.
  - 8 near-duplicate coordinate pairs (<5m apart, different codes) — almost certainly the same physical site logged/reclassified twice. Each pair was resolved by keeping one site and logging the dropped one.
- **Regional grouping:** a 3-letter code-prefix (e.g. `acr`, `ama`, `ron`) maps informally to Brazilian states (Acre, Amazonas, Rondônia) and is retained as a grouping variable for future stratified/grouped cross-validation (`StratifiedGroupKFold`), to prevent same-site leakage across folds.

**Output:** `amazon_geoglyphs_cleaned.csv` — **1,443 sites**, plus `duplicate_resolution_log.csv` documenting every dropped near-duplicate and why.

## 4. Imagery source and currency decision

Two imagery options were considered for downloading positive images:
- Official Google Maps Platform Static API (licensed, citable, but requires an API key/billing)
- Unofficial `mt0.google.com/vt` tile-stitching endpoint (no key, no license — used by the prior team's `v2gmapDownloader.py`)

**Decision (interim):** continue with the unofficial endpoint for now, since it requires no setup and worked reliably in testing. **Flagged as an open provenance concern to revisit before publication** — a reviewer may reasonably ask about the imagery license/citation.

A separate, important finding surfaced during testing (see Section 6): the catalog's `source` column (e.g. "GE 2021-11-20") indicates specific historical Google Earth imagery dates when each site was known to be visible — likely during a temporary forest-clearing event. The tile endpoint used here has **no date control** and serves whatever imagery is currently cached, which does not guarantee the same clearing is still visible today.

## 5. Positive-image downloader: design

Built `download_positives.py`, rewriting the prior team's `v2gmapDownloader.py` to fix several issues:

| Issue in original script | Fix |
|---|---|
| Fixed zoom=20 for every site regardless of size | **Adaptive zoom per site**, computed from the site's actual dimensions (`a_width`/`b_width`) |
| No cap on window size | Oversized linear "zanja" earthworks (up to 2,100m long) would otherwise demand a ~6,300m window at low resolution. Added `MAX_WINDOW_M = 800`, capping the crop to a detailed, recognizable segment rather than the full length at low detail |
| No HTTP status checking | Failed/blank tiles are now detected and logged, not silently stitched in |
| No throttling | Added per-tile and per-site delays to avoid triggering rate limits/blocking (1,443 sites × 64 tiles ≈ 92,000 requests) |
| Not resumable | Manifest-CSV-based resume: already-downloaded sites are skipped on rerun |
| Grayscale conversion by default | Dropped — no justification for discarding 2 of 3 color channels before training a classifier |

Tile grid stays fixed at 8×8 (256px tiles), matching the prior team's `V2segmentationMaskGenerator.py` assumption — only the zoom (ground coverage) varies per site.

## 6. Positive-image download: execution and results

Run in Google Colab, cleaned CSV as input, output written directly to Google Drive (`geoglyph_v2/positives/`) so progress survives Colab disconnects.

- **Result: 1,436 of 1,443 sites downloaded successfully** (99.5%); 7 sites returned too many missing tiles.
- The run disconnected partway through (~1,250/1,443) and resumed cleanly from the Drive-backed manifest with no lost progress or duplicate work.
- **Manifest verification:** row counts in the manifest matched the actual file count on disk exactly (1,436 = 1,436) — no orphaned entries either way.
- **Visual spot-check (n=9, random sample):** most sites showed a clearly visible earthwork. One notable exception (`abuc1`, a 203m circle) showed only forest canopy — no visible geoglyph — despite a technically clean download (0 missing tiles, correctly centered). This is consistent with the imagery-currency issue in Section 4: the site's defining clearing event (2021) may have since regrown.

**Implication carried forward:** a successful download does not guarantee a visually confirmed positive. A systematic verification step is still needed before the full 1,436-image set can be trusted as ground truth (see Section 7 — the prior team hit this exact problem).

## 7. Resolved: how the prior team reduced ~1,436 candidates to 530 positives

This was an open question for weeks: why did Zach and Colin's final positive set (530 images) contain so few images relative to the ~1,400+ candidate sites in the catalog they worked from? Their uploaded code (`DataPreprocessingByZachnColin/`) resolves this directly.

**Finding: the reduction was manual, image-by-image visual curation — not a coordinate or metadata filter.**

- `dataFilter.py` is a Tkinter GUI tool that displays every downloaded candidate image full-screen, one at a time. A human presses **Right Arrow = Keep, Left Arrow = Discard** while looking at it. Every discard is logged by ID to a `<shape>_bad_image_ids.txt` file.
- `removeIds.py` reads all such bad-ID files and deletes the corresponding images from the working dataset.
- `strictly_filtered_bad_geoglyph_ids.txt` is the consolidated result of this process: **599 images manually flagged and discarded** as not showing a usable/visible geoglyph.

**Independent confirmation:** the very first entry in their 599-item discard list is `geoglyph_circle_#abuc1` — the exact same site independently found in Section 6 to show no visible earthwork in current imagery. The prior team encountered the identical imagery-visibility problem and solved it the same way we concluded was necessary: a human visibility check after automated download, not before.

**Negative-sampling logic also confirmed by code** (`negativeGenerator.py`): each negative is generated by offsetting a known site's coordinate by up to ~0.003° (~330m), while attempting to stay at least ~111m (`0.001° × 111 km/°`) from any known site — i.e., a deliberate "hard negative, near-miss" design, exactly as reverse-engineered independently from the coordinate-proximity audit in Section 8. Their intended 111m minimum clearance is looser than several of the contaminated negatives found in that audit (down to 67m), suggesting a gap in their own distance-rejection logic likely explains how those contaminated negatives were generated in the first place.

**Practical implication:** our own plan (automated coordinate/proximity checks + targeted manual review of small flagged subsets, rather than full manual review of 1,443 images) is a direct, more scalable evolution of the same verification principle the prior team used, informed by having seen where their fully-manual approach still let contamination through.

## 8. Negative-image dataset: investigation (in progress)

Rather than generating a new negative set immediately, the existing 2,181 "non-geoglyph" images (from the original Zach/Colin dataset) were audited for reuse.

**Contamination check, attempt 1 (filename code matching):** failed — negative filenames follow the pattern `negative_<code>_<lat>_<lon>.png`, and a naive regex extracted the literal word "negative" instead of the embedded site code. Result: 0 flagged, which was correctly treated as unreliable rather than trusted at face value.

**Contamination check, attempt 2 (coordinate proximity):** parsed the actual lat/lon embedded in each negative's filename and computed the distance to the nearest official catalog site (BallTree, haversine).

**Key finding:** distances ranged from 67m to 484m across all 2,181 negatives (mean 265m), and most negatives' nearest official site matched the same code embedded in their own filename. This confirmed the prior team's negative-sampling strategy (see Section 7) was a deliberate hard-negative / near-miss design, not random background sampling.

This raises two open issues:
1. **38 negatives sit within 100m of a known site** — close enough, relative to some sites' actual size (a_width 100–200m+), that the real earthwork may bleed into the capture window. Visual inspection of these 38 confirmed **some genuinely contain a visible geoglyph** — i.e., real mislabeling, caught only by the coordinate-based check after the filename-based check missed it entirely.
2. **The existing negative set may have zero examples of plain, generic background** far from any known site — every one of the 2,181 is within 484m of a real geoglyph. If reused as-is, the model would never see an example of "ordinary forest with nothing nearby," which could limit generalization.

**Status: manual removal of confirmed-contaminated images from the 38 flagged candidates is in progress** (user reviewing directly in the Google Drive folder).

## 9. Open design considerations flagged for later

- **Ground-coverage matching:** positives use a variable window (150–800m, adaptive to site size). If negatives are captured at a different, less-varied window distribution, the model could learn to distinguish classes by zoom level/resolution rather than actual visual content (a shortcut-learning risk). Negative capture windows should be drawn from the same distribution as positives.
- **Negative diversity:** the reused hard-negative set (near-miss samples) should likely be supplemented with genuinely generic background negatives (plain forest, deforested-but-empty land) to avoid the gap noted in Section 8.
- **Pixel dimensions vs. ground coverage:** all images must be resized to one fixed pixel size at training time (architecture-dependent), which is independent of and does not resolve the ground-coverage-matching issue above.
- **Verification-before-labeling:** given the imagery-currency finding (Section 6/7) and the negative-mislabeling finding (Section 8), no downloaded image — positive or negative — should be treated as a confirmed label without some verification step. Manual review at full scale (thousands of images) was explicitly ruled out as infeasible; automated/coordinate-based checks plus targeted manual review of small flagged subsets is the adopted approach going forward, directly informed by how the prior team's fully-manual approach still let some contamination through (Section 7).
- **Datum-shift risk (Section 2):** cannot be resolved from catalog metadata alone; worth stating as a documented limitation in the thesis rather than something to fully solve.
- **Framing for the paper:** the model is intended to be recall-optimized with a human-in-the-loop triage framing (favor flagging possible sites for human review over missing real ones), which supports treating dataset construction itself — including the verification methodology — as part of the methodological contribution, not just a preprocessing step.

## 10. Current dataset state (as of this log)

| Item | Count | Status |
|---|---|---|
| Official catalog sites (raw) | 1,477 | — |
| Valid-coordinate sites | 1,455 | — |
| Cleaned sites (post dedup) | 1,443 | Final — `amazon_geoglyphs_cleaned.csv` |
| Positive images downloaded | 1,436 / 1,443 | Downloaded, spot-checked; full visual verification still pending |
| Existing negative images | 2,181 | Under audit; 38 flagged near known sites, manual removal in progress |
| Prior team's discarded positives (reference) | 599 | Manually flagged as not visible/usable — informs our own verification design |

## 11. Next steps

1. Finish manual removal of confirmed-contaminated images from the 38 flagged negatives; re-verify folder count afterward.
2. Decide how to handle the negative-diversity gap (Section 8, item 2) — supplement with newly generated generic-background negatives.
3. Address ground-coverage/zoom matching between positives and negatives (Section 9) before finalizing the negative set.
4. Design a systematic (partially automated) visibility-verification step for the positive set, informed by the prior team's manual keep/discard approach (Section 7) but scaled beyond full manual review.
5. Assemble a single manifest CSV (image path, label, site code, zoom/window, region group) covering both classes for reproducible, leakage-free model training and cross-validation.
6. Document the datum-shift and catalog-duplication risks (Section 2) explicitly as limitations in the thesis methodology section.

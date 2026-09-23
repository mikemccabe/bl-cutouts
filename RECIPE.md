# British Library book images - access notes

1,080,814 illustrations cut from 49,455 digitised books (c. 1510-1900), released into the public
domain by British Library Labs. As of August 2026 every image in the `embellishments`, `plates` and
`medium` configs also carries a model-predicted instance mask, so each illustration can be lifted
off its page as a clean cutout. There are no subject tags; retrieval is by SigLIP2 vector
similarity.

## What is on the Hub

`biglam/british-library-book-images` (CC0). Image configs, split by an algorithmic guess at what
each crop is: `embellishments` (416,935), `plates` (385,231), `medium` (217,100), `covers` (61,548).
Fields: `image`, `date` (string; 0.5% are "Unknown"), `fname`, `image_type`. Full images ~621 GB.

Derived configs:

- `siglip2_embeddings` - one 1152-d float32 vector per image from
  `google/siglip2-so400m-patch16-256`, row-aligned with the image configs. Dual encoder, so text
  queries embed into the same space; rank by cosine. ~5 GB for the full matrix.
- `crop_masks` - 3,022,916 instances over 1,019,266 images, one row per image, joinable on `fname`.
  Fields `objects` (`bbox` as [x,y,w,h], `score`, `area`, `rectangularity`) and `masks_rle`, a JSON
  string of COCO RLE dicts in the full source frame. 3.96 GB. `covers` excluded (the model cannot
  abstain).

Scores were kept down to 0.10 so consumers pick their own operating point; 59.7% of instances fall
below 0.3. Use `score >= 0.3` for display. 76% of images have exactly one instance. Masks are
predictions, not hand-checked (97.4% judged acceptable on a 40-image human check).

## 1. Vector search over the web (no download)

Space: https://huggingface.co/spaces/davanstrien/historical-illustration-search
Base URL: https://davanstrien-historical-illustration-search.hf.space
CORS is reflected for any origin, including `null`, so a static HTML page can call it directly.

The Space also indexes 411,385 figure crops from the
Encyclopaedia Britannica, 1,492,199 images in all. `/search_collections` takes a `dataset`
argument (`all`, `bl`, `britannica`); the older endpoints search both.

```bash
B=https://davanstrien-historical-illustration-search.hf.space
EID=$(curl -s -X POST "$B/gradio_api/call/search_collections" \
  -H 'content-type: application/json' \
  -d '{"data":["a platypus","all",0,0,8,[],[],0.5,"bl"]}' | jq -r .event_id)
curl -s -N "$B/gradio_api/call/search_collections/$EID"
```

Endpoints:

- `/search_collections` - query, collection, year_from, year_to, k, positive_rows,
  negative_rows, text_weight, dataset
- `/search` - query, collection, year_from, year_to, k
- `/search_reference` - adds `seed_row`, `text_weight` (blend a phrase with an example image)
- `/search_feedback` - adds `positive_rows`, `negative_rows` (relevance feedback)

Schema: `/gradio_api/info` and `/openapi.json`. Results carry `row_id`, `asset_id`, `dataset`,
`thumb`, `full`, `title`, `year`, `collection`, `cutout`, `score`. For BL rows `thumb` and `full`
are bucket URLs and `cutout` is a Space path; Britannica rows return all three as Space paths
(`/illustration/...`). `row_id` is a temporary index reference; `asset_id` is the durable one.
Near-identical results (cosine >= 0.95) are collapsed, so a query can return fewer than `k` rows.
Typical latency 1.5-2.5 s warm.

## 2. Single images and cutouts by URL

```
https://huggingface.co/buckets/biglam/bl-images/resolve/{thumbs|full}/{split}/{fname}
{space}/cutout?fname={fname}&size={thumb|full}    # masked PNG, composited on demand
```

`size=full` returns the native-resolution cutout (a few MB), cached on the Space's disk.
Outside-mask pixels are painted white, not transparent. That region is border-connected, so a flood
fill from the frame converts exactly it to alpha if transparency is needed.

## 3. SQL against the parquet, in place

DuckDB reads the Hub directly with column pruning - no image bytes move unless `image` is selected.
A cold run of this query, install included, takes about five seconds.

```sql
SELECT i.fname, i.date, m.objects, m.masks_rle
FROM 'hf://datasets/biglam/british-library-book-images/plates/*.parquet' i
JOIN 'hf://datasets/biglam/british-library-book-images/crop-masks/plates-*.parquet' m
USING (fname)
```

Decoding a mask:

```python
import json
from datasets import load_dataset
from pycocotools import mask as maskutil

masks = load_dataset("biglam/british-library-book-images", "crop_masks", split="plates")
row = masks[0]
rles = json.loads(row["masks_rle"])
keep = [i for i, s in enumerate(row["objects"]["score"]) if s >= 0.3]
m = maskutil.decode(rles[keep[0]])   # HxW numpy array, source frame
```

The leading digits of `fname` are the British Library system number, which is also the `record_id`
of the OCR corpus below.

## 4. The page text of the same books

The same digitisation programme released its OCR, and it is full text rather than metadata or
snippets: `biglam/blbooks-parquet`, 14,011,953 rows (one per page), ~30 GB, CC0. Fields: `text`,
`pg`, `empty_pg`, `mean_wc_ocr` and `std_wc_ocr` (per-page OCR word confidence), `record_id`, plus
catalogue metadata - `title`, `name`, `all names`, `place`, `Publisher`, `date`, `raw_date`,
`Country of publication`, `Language_1..4`, `multi_language`. Languages: en, de, es, fr, it, nl.
Files at `hf://datasets/biglam/blbooks-parquet/data/train-*.parquet`.

```sql
-- the text of the book an illustration was cut from
SELECT pg, text
FROM 'hf://datasets/biglam/blbooks-parquet/data/train-*.parquet'
WHERE record_id = '002253078'      -- leading digits of the image's fname
  AND NOT empty_pg
  AND mean_wc_ocr > 0.8            -- drop the pages OCR made a mess of
ORDER BY pg
```

Joining images to text:

```python
fname = "002253078_02_000165_1_The Illustrated London Reading Books...jpg"
record_id = fname.split("_")[0]    # "002253078"
```

**The join is book-level, not page-level.** `fname` encodes a page position, but it is not
guaranteed to align with the `pg` column, so verify page-level alignment rather than assuming it.
96% of system numbers matched in a 20k-row sample; the other 4% are books in the image release but
absent from the OCR release.

Caveats specific to this corpus:

- Raw OCR of 19th-century print with no correction pass. Confidence below 0.5 is common and often
  unusable - that is what `mean_wc_ocr` is for. A spot check of a 1690s play returned pages at
  0.34-0.51.
- Metadata is a 2021 catalogue export: several fields are unpopulated much of the time, and the
  language fields were partly determined computationally.

## Gotcha: go easy on the cutout endpoint

The Space composites BL cutouts on a shared CPU, two decodes at a time. Run 2 requests at a time
with exponential backoff, and retry a failure before trusting it.

## Caveats

- `date` is a string; 5,291 rows are "Unknown" and 2,151 carry post-1900 dates that are catalogue
  errors.
- The corpus is overwhelmingly late-Victorian: the 1890s alone are a third of it. Treat it as a
  record of what one institution digitised, not a sample of printed illustration.
- Public domain (no known copyright restrictions), tagged CC0 on the Hub. Attribution to the
  British Library is expected practice, not a legal condition. Nothing has been reviewed for
  offensive content; colonial-era publishing is heavily represented.

## Announcement

https://www.linkedin.com/feed/update/urn:li:activity:7496149798677557248/

## Sources

- Dataset: https://huggingface.co/datasets/biglam/british-library-book-images
- Search Space: https://huggingface.co/spaces/davanstrien/historical-illustration-search
- Mask model: https://huggingface.co/davanstrien/bl-crop-tighten-rfdetrseg-clip10
- OCR text: https://huggingface.co/datasets/biglam/blbooks-parquet

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

Space: https://huggingface.co/spaces/davanstrien/bl-images-search
Base URL: https://davanstrien-bl-images-search.hf.space
CORS is reflected for any origin, including `null`, so a static HTML page can call it directly.

```bash
B=https://davanstrien-bl-images-search.hf.space
EID=$(curl -s -X POST "$B/gradio_api/call/search" \
  -H 'content-type: application/json' \
  -d '{"data":["a platypus","all",0,0,8]}' | jq -r .event_id)
curl -s -N "$B/gradio_api/call/search/$EID"
```

Endpoints:

- `/search` - query, collection, year_from, year_to, k
- `/search_reference` - adds `seed_row`, `text_weight` (blend a phrase with an example image)
- `/search_feedback` - adds `positive_rows`, `negative_rows` (relevance feedback)

Schema: `/gradio_api/info` and `/openapi.json`. Results carry `row_id`, `thumb`, `full`, `title`,
`year`, `collection`, `cutout`. Typical latency ~0.5 s.

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

The leading digits of `fname` are the British Library system number = `record_id` in
`biglam/blbooks-parquet` (14,011,953 pages of OCR text from the same programme). 96% of system
numbers match in a 20k-row sample. The join is book-level, not page-level.

## Gotcha: the cutout endpoint is not concurrency-safe

16 parallel cutout requests returned a mix of `404 no mask for this image` and
`502 thumb fetch failed`; the same 36 URLs fetched sequentially returned 36x200. The 404 comes from
a shared DuckDB handle used across threads - `fetchone()` returns None and a present mask is
reported missing. Nothing is actually absent. Serialise, or run 2 at a time with exponential
backoff, and retry rather than trusting the status code.

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
- Search Space: https://huggingface.co/spaces/davanstrien/bl-images-search
- Mask model: https://huggingface.co/davanstrien/bl-crop-tighten-rfdetrseg-clip10
- OCR text: https://huggingface.co/datasets/biglam/blbooks-parquet

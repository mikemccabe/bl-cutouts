# A Million Cutout Engravings

### -> Live page: https://mikemccabe.github.io/bl-cutouts/

Search the collection by describing a picture, and pull any result out as a clean cutout.

A one-page report on the British Library's 1,080,814 public-domain book illustrations, which as of
August 2026 carry model-predicted masks -- so each illustration can be lifted off its scanned page
as a clean cutout -- and which are searchable by description rather than by tags.

The page is static and runs its search live in the browser against the public search Space; there is
no server component and nothing is downloaded in advance.

- `index.html` -- the report; seeded query is "a platypus"
- `RECIPE.md` -- the access recipes on their own, for pasting into a notebook or a language model
- `fonts/` -- Latin Modern (the OpenType Computer Modern), vendored so the page has no CDN dependency

## Sources

- Dataset: https://huggingface.co/datasets/biglam/british-library-book-images
- Search Space: https://huggingface.co/spaces/davanstrien/bl-images-search
- Mask model: https://huggingface.co/davanstrien/bl-crop-tighten-rfdetrseg-clip10
- OCR text of the same books: https://huggingface.co/datasets/biglam/blbooks-parquet
- Announcement: https://www.linkedin.com/feed/update/urn:li:activity:7496149798677557248/

## Licence

The illustrations are public domain (no known copyright restrictions); attribution to the British
Library is expected practice rather than a legal condition. Latin Modern is under the GUST Font
License. The page itself is MIT.

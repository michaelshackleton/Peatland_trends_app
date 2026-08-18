# vendor/

Verbatim copies of the files this app needs from the parent `rs-utils-main`
repository's `src/` folder (the same files vendored into `Satellite
downloader/vendor/` and `Claude_script/vendor/`), copied in so that
`Peatland_trends` is fully self-contained and can be moved or copied to any
location without needing the rest of `rs-utils-main` alongside it.

Copied unmodified from `Satellite downloader/vendor/`:

- `rs_data.py` - `STACDataManager`, `STACRSRegionManager`
- `rs_processing.py` - `RSDataProductManager`
- `utils.py` - used by `rs_data.py` for rasterising vector layers (also used
  directly by `modules/analysis.py` here to build the polygon pixel mask
  used for cloud/coverage stats and the per-pixel mean raster)
- `rsderiv/__init__.py`, `rsderiv/sharpen.py`, `rsderiv/sen1.py` - used by
  `rs_processing.py` for pan-sharpening (Sentinel-2's SWIR bands) and
  (unused here, but imported regardless) Sentinel-1 speckle filtering

If the original scripts in `src/` are updated with bug fixes or new
features you want here, these vendored copies won't pick that up
automatically - they'd need to be re-copied by hand.

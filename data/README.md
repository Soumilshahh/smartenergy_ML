# Dataset

Run `python -m src.data.download_data`. The official UCI ZIP URL is recorded in the downloader. Raw CSV and retrieval provenance live in ignored `data/raw/`; public provenance is copied into experiment metadata.

19735 rows, 29 columns. 2016-01-11 17:00:00 to 2016-05-27 18:00:00. Measured Wh per ten-minute interval; household temperature/humidity and merged airport weather. Primary models omit airport interpolation, random controls and lights. No observations were deleted. The first 144 origins provide history; the final six cannot form complete labels.

CSV SHA-256: `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`. Verify this hash before comparing a different download with the reported artifacts. Source: https://archive.ics.uci.edu/dataset/374/appliances-energy-prediction. DOI 10.24432/C5VC8G. License CC BY 4.0.

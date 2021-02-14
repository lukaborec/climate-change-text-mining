`python run.py`

with following arguments:

`--method` ("glove", "tfidf") — which method to use for ranking of the articles. Required.

`--glossary` ("normal", "enriched") — for loading glossary of cc words. Optional, defaults to "normal".

`--length` (float in range [0,1]) — how much of the article to keep. Optional, defaults to 0.5.

`--frame` (0 or 1) — whether to use frame-specific information for ranking or not. Optional, defaults to 0.


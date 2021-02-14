Project Module Mining Opinions and Arguments Wintersemester 2020/2021

This project builds on top of work from [Hulme et al](https://www.nature.com/articles/s41558-018-0174-1?WT.ec_id=NCLIMATE-201806&spMailingID=56720253&spUserID=ODE0MzAwNjg5MAS2&spJobID=1405001778&spReportId=MTQwNTAwMTc3OAS2). We look into text mining techniques in order to explore how the framing and sentiment towards climate change have changed in the past 60 years. We analyze a corpus of ~500 articles about climate change that were published in Nature and Science journals.


### Running the script
`python run.py`

with following arguments:

`--method` ("glove", "tfidf") — which method to use for ranking of the articles. Required.

`--glossary` ("normal", "enriched") — for loading glossary of cc words. Optional, defaults to "normal".

`--length` (float in range [0,1]) — how much of the article to keep. Optional, defaults to 0.5.

`--frame` (0 or 1) — whether to use frame-specific information for ranking or not. Optional, defaults to 0.


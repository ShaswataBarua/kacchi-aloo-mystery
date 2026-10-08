The Great Kacchi Aloo Mystery

Predicting whether wedding guests go back for a second plate of kacchi biryani, for the Bangladesh AI Olympiad (BdAIO) World AI Week 2026 competition on Kaggle.

Public leaderboard score: 0.95027 accuracy

Key finding

Guests go back for seconds only when there is enough aloo for each person, but not too much. The total number of potatoes does not matter. The one number that does is aloo per guest (aloo_count / guests).

Aloo per guest	Share of weddings with seconds
below 0.9	0%
1.0 to 1.1	81%
1.2 to 1.8	97% to 100%
2.0 to 2.1	27%
above 2.2	about 0%
Below about 1 potato per guest: people feel there was not enough, so nobody goes back.
About 1 to 2 potatoes per guest: guests are satisfied and go back.
Above about 2 potatoes per guest: too much aloo suggests less mutton, so guests lose trust in the biryani.

Because the rule uses a ratio, it does not depend on wedding size. This matters because the test set contains weddings larger than anything in the training set, where models built on raw counts fail.

Approach
Data cleaning
Converted Bangla digits in fairy_lights to English digits.
Excluded training rows with a missing aloo_count instead of guessing a value.
Corrected Tutul's extra-zero errors: counts with an aloo-per-guest ratio above 6 were divided by 10.
Feature design: aloo per guest, which is independent of wedding size.
Model: a small decision tree on aloo per guest. It learns a band of roughly 0.97 to 2.01 potatoes per guest.
Other features tested: mutton, borhani, fairy lights, dhol players, aunties, drone, event and city. None improved cross-validated accuracy, so the final model does not use them.
Results
Model	5-fold CV accuracy (train)
Decision tree, depth 2, aloo per guest only	about 95.0%
Decision tree, depth 3, aloo per guest only	about 95.6%
Gradient boosting on log counts	about 87%
Random forest on raw counts (starter baseline)	about 77% (single hold-out split)

Public leaderboard: 0.95027.

Limitations

Most remaining errors are weddings with about 0.94 to 1.15 or 1.7 to 2.25 potatoes per guest. Near these edges the recorded potato counts appear to be slightly inaccurate, and none of the other columns helped separate those cases. The data is synthetic, so the findings describe this dataset and not real weddings.

Repository contents
File	Description
aloo-theory-kacchi-mystery.ipynb	Kaggle notebook with the analysis and the aloo theory
kacchi_v2.py	Standalone script that reproduces the final model
README.md	This file
How to run
Open the competition on Kaggle and create a notebook with the competition data attached.
Paste in or upload the notebook code (or kacchi_v2.py) and click Run All.
The script reads train.csv and test.csv from /kaggle/input and writes submission.csv.

To run it locally, download train.csv and test.csv from the competition's Data tab, place them in the same folder as the script, and run:

bash
pip install pandas numpy scikit-learn
python kacchi_v2.py
Data

The competition data is not included in this repository. Download it from the competition page and follow the competition rules.

Acknowledgements

Competition by Tasnim Mahfuz Nafis and the Bangladesh AI Olympiad (BdAIO) for World AI Week 2026. This project was developed with assistance from Claude (Anthropic).

Author

Shaswata, github.com/ShaswataBarua

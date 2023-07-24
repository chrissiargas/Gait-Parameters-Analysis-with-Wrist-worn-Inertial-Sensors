import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = 'Tan'
archive = os.path.join("archive", model)
path = os.path.join(archive, "save-" + '20230723-202312')
cross_val = 'start'
enviroment = 'outdoors'
hparams = 'split-' + cross_val + '-enviroment-' + enviroment
path = os.path.join(path, hparams)
scores_file = os.path.join(path, "scores.csv")

results = pd.read_csv(scores_file)
results['event_'] = results['event'].str[-2:]

plt.figure()
sns.set_context("notebook", font_scale=1.3)
g = sns.boxplot(x="activity", y="f1 score",
                hue="event_",
                data=results)
g.set_xticklabels(g.get_xticklabels(), rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show()

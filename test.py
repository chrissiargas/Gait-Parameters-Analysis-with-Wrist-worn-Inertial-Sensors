
from dataset import Dataset

D = Dataset(regenerate=False)
train, val, test = D(batch_prefetch=False, time_info=True)

for i, example in enumerate(train.take(3)):

    data = example[0]
    y = example[1]
    XTime, yTime = example[2]
    print(data)
    print(y)
    print(XTime)
    print(yTime)




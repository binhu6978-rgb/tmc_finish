"""Frozen CIFAR100 training split and stratified diagnostic/evaluation test split."""
import json
from pathlib import Path
import statistics
import numpy as np
from torchvision import datasets,transforms

SPLIT='cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'


def partition(root):
    d=json.loads((Path(root)/'data/splits'/SPLIT).read_text(encoding='utf-8'))
    users={int(k):v for k,v in d['train_data'].items()}
    n=[len(users[i]) for i in range(100)];indices=[v for a in users.values() for v in a]
    if (min(n),statistics.median(n),max(n),sum(n))!=(262,491.5,757,50000):
        raise ValueError('wrong cached split; never regenerate it')
    if sorted(indices)!=list(range(50000)):
        raise ValueError('training indices overlap or missing')
    return users


def load_data(root):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
        (.5071,.4867,.4408),(.2675,.2565,.2761))])
    # Explicitly disable downloads and all train transforms beyond normalization.
    return (datasets.CIFAR100(root,train=True,download=False,transform=transform),
            datasets.CIFAR100(root,train=False,download=False,transform=transform))


def evaluation_split(targets,seed=20261017):
    rng=np.random.RandomState(seed);y=np.array(targets);val=[];reserved=[]
    for c in range(100):
        ids=np.where(y==c)[0];rng.shuffle(ids)
        if len(ids)!=100:raise ValueError('expected 100 official test images per class')
        val.extend(ids[:50].tolist());reserved.extend(ids[50:].tolist())
    # Mix classes once: BN batch statistics depend on ordering. Fixed thereafter.
    rng.shuffle(val);rng.shuffle(reserved)
    assert not set(val)&set(reserved)
    return dict(seed=seed,validation=val,reserved=reserved,each_class_per_subset=50,
                training_source='official train',diagnostic_source='official test',
                reserved_is_historically_untouched=False)

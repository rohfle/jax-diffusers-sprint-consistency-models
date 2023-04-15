import torch
torch.set_printoptions(precision=4, linewidth=140, sci_mode=False)
torch.manual_seed(1)

# from miniai.imports import *
# from miniai.diffusion import *
from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
# from torch import F, TF

n_steps = 1000
BATCH_SIZE = 512
xl,yl = 'image','label'

dsd = load_dataset('fashion_mnist')

def collate_consistency(b):
    return default_collate(b)['image']

def dl_consistency(ds):
    return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_consistency, num_workers=8)

# stuff from miniai to get rid of
def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f

class itemgetter:
    """
    Return a callable object that fetches the given item(s) from its operand.
    After f = itemgetter(2), the call f(r) returns r[2].
    After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])
    """
    __slots__ = ('_items', '_call')

    def __init__(self, item, *items):
        if not items:
            self._items = (item,)
            def func(obj):
                return obj[item]
            self._call = func
        else:
            self._items = items = (item,) + items
            def func(obj):
                return tuple(obj[i] for i in items)
            self._call = func

    def __call__(self, obj):
        return self._call(obj)

    def __repr__(self):
        return '%s.%s(%s)' % (self.__class__.__module__,
                              self.__class__.__name__,
                              ', '.join(map(repr, self._items)))

    def __reduce__(self):
        return self.__class__, self._items

def collate_dict(ds):
    get = itemgetter(*ds.features)
    def _f(b): return get(default_collate(b))
    return _f

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))


class DataLoaders:
    def __init__(self, *dls): self.train,self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        f = collate_dict(dd['train'])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))


@inplace
# this is a common way to normalize image data in pytorch
# .pad() is adding 2 rows / columns of zeroes to top, right, bottom, left
# then *2-1 is shifting range of pixel values from -1 to 1
def transformi(b):
    b['image'] = [
        F.pad(
            TF.to_tensor(o),
            (2,2,2,2)
        )*2-1 for o in b['image']
    ]

# as above dsd is the imported dataset
tds = dsd.with_transform(transformi)
# splitting dataset into different training and test sets
dls = DataLoaders(dl_consistency(tds['train']), dl_consistency(tds['test']))



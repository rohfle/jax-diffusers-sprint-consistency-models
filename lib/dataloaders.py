
def collate_consistency(b):
    return default_collate(b)[xl]
def dl_consistency(ds):
    return DataLoader(ds, batch_size=bs, collate_fn=collate_consistency, num_workers=8)

@inplace
# this is a common way to normalize image data in pytorch
# .pad() is adding 2 rows / columns of zeroes to top, right, bottom, left
# then *2-1 is shifting range of pixel values from -1 to 1
def transformi(b):
    b[xl] = [F.pad(TF.to_tensor(o), (2,2,2,2))*2-1 for o in b[xl]]

# as above dsd is the imported dataset
tds = dsd.with_transform(transformi)
# splitting dataset into different training and test sets
dls = DataLoaders(dl_consistency(tds['train']), dl_consistency(tds['test']))


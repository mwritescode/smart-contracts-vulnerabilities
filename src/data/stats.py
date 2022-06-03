import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

class GetMeanStd:
    """
    Inspired by the implementation of https://github.com/Nikronic/CoarseNet/blob/master/utils/preprocess.py#L142-L200
    """
    def __init__(self, dataset, batch_size, img_size):
        self.img_transform =transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])

        dataset.set_transform(self.__to_tensor)
        self.data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=0)
    
    def __to_tensor(self, examples):
        examples['image'] = [self.img_transform(elem) for elem in examples['image']]
        return examples

    def __call__(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data in tqdm.tqdm(self.data_loader, desc='Computing stats..'):
            data = data['image']
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        return mean, std
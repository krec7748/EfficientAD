from torchvision.datasets import ImageFolder

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, targets = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path
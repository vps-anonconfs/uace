import abc


class Skeleton(abc.ABC):
    def get_train_dataset(self):
        raise NotImplementedError()

    def get_test_dataset(self):
        raise NotImplementedError()

    def num_classes(self):
        raise NotImplementedError()
        
    @property
    def concept_names(self):
        raise NotImplementedError()
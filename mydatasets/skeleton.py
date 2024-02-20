import abc


class Skeleton(abc.ABC):
    def get_train_dataset(self):
        # this routine is useful if we need to train a model using a train split and compute explanations using test split
        raise NotImplementedError()

    def get_test_dataset(self):
        # usually the split that is used to compute explanations.
        raise NotImplementedError()

    def num_classes(self):
        # this is required to inform the trainer of the number of classes. Need not be set if there is no training.
        raise NotImplementedError()
        
    @property
    def concept_names(self):
        raise NotImplementedError()
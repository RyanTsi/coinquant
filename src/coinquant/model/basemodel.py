import abc

class BaseModel(abc.ABC):

    @abc.abstractmethod
    def fit(self, train_loader, valid_loader):
        pass

    @abc.abstractmethod
    def predict(self, data_loader):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
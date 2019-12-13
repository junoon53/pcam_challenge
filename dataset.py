import h5py
from torch.utils import data

class H5Dataset(data.Dataset):
    
    def __init__(self, x_archive, y_archive, transform=None):
          self.x_archive = h5py.File(x_archive, 'r')
          self.y_archive = h5py.File(y_archive, 'r')
          self.labels = self.y_archive['y']
          #self.labels = (self.labels[0], self.labels[3], self.labels[1], self.labels[2])
          self.data = self.x_archive['x']
          #self.data = (self.data[0], self.labels[3], self.labels[1], self.labels[2])
          self.transform = transform
            
          print(type(self.data), self.data.shape, self.labels.shape)

    def __getitem__(self, index):
        
        datum = self.data[index]
        
        # print("before", datum)

        if self.transform is not None:
            datum = self.transform(datum)
            # datum = (datum)/255

        # print("after", datum)
        
        return datum, self.labels[index]

    def __len__(self):
        
        return len(self.labels)

    def close(self):
        self.archive.close()

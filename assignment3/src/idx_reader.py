import numpy as np


class IdxReader:
    def __init__(self, filename, format):
        with open(filename, "rb") as f:
            b_magic_number = f.read(4)
            b_count = f.read(4)

            # Get actual magic number and count
            self.magic_number = int.from_bytes(b_magic_number, byteorder='big')
            self.count = int.from_bytes(b_count, byteorder='big')

            # Print out for debugging
            print(self.magic_number)
            print(self.count)

            # Data handled differently for these
            if format == 'double':
                b_dimensions = f.read(4)
                self.dimensions = int.from_bytes(b_dimensions, byteorder='big')
                dtype = np.dtype('float64').newbyteorder('>')
            elif format == 'ubyte':
                dtype = np.dtype(np.ubyte)
                self.dimensions = 1
            elif format == 'bitmap':
                b_dimensions = f.read(8)
                self.dimensions = 784
                dtype = np.dtype(np.ubyte)

            print(self.dimensions)

            self.data = np.fromfile(f, dtype=dtype)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= self.count:
            raise StopIteration
        else:
            data_index = self.iter_index * self.dimensions
            self.iter_index += 1
            data = self.data[data_index:data_index + self.dimensions]

            if self.dimensions == 1:
                arr_data = np.zeros((10, 1))
                arr_data[data[0]] = 1.0
                data = arr_data
            return data

    def __getitem__(self, item):
        if item < self.count:
            data_index = item * self.dimensions
            data = self.data[data_index:data_index + self.dimensions]

            if self.dimensions == 1:
                arr_data = np.zeros((10, 1))
                arr_data[data[0]] = 1.0
                data = arr_data
            return data
        else:
            raise IndexError

    def __len__(self):
        return self.count


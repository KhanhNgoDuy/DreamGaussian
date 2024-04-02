from pywavefront import Wavefront
import numpy as np
from pathlib import Path


class Mesh:
    def __init__(self, obj_file):
        self.obj_file = obj_file
        
        self.verts = np.array(self.extract_mesh())
        self.mean = np.mean(self.verts, axis=0)
        self.max = np.amax(self.verts, axis=0)
        self.min = np.amin(self.verts, axis=0)
        
        self.vals = [self.mean, self.max, self.min]
        
        # normalization makes the vertices surround point (0, 0, 0)
        self.verts = self.normalize(self.verts)

    def extract_mesh(self):
        vertices = Wavefront(self.obj_file, create_materials=True).vertices
        return vertices
        
    def normalize(self, verts):
        offset = (self.max + self.min) / 2
        verts = verts - offset
        return verts


def to_float(ls):
    result = []
    for val in ls:
        result.append(float(val))
    return result


def plot_1D(verts):        
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.hist(x, bins=50, color='r')
    plt.title('Distribution of X coordinates')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 1, 2)
    plt.hist(y, bins=50, color='g')
    plt.title('Distribution of Y coordinates')
    plt.xlabel('Y')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 1, 3)
    plt.hist(z, bins=50, color='b')
    plt.title('Distribution of Z coordinates')
    plt.xlabel('Z')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    

def plot_2D(verts):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY plane
    axs[0].scatter(x, y, s=5)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('XY Plane')
    
    # XZ plane
    axs[1].scatter(x, z, s=5)
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Z')
    axs[1].set_title('XZ Plane')
    
    # YZ plane
    axs[2].scatter(y, z, s=5)
    axs[2].set_xlabel('Y')
    axs[2].set_ylabel('Z')
    axs[2].set_title('YZ Plane')
    
    plt.tight_layout()
    plt.show()


def normalize(verts):
    max = np.amax(verts, axis=0)
    min = np.amin(verts, axis=0)
    offset = (max + min) / 2
    verts = verts - offset
    return verts


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    logs = './logs'
    templates = './templates'
    
    obj_list = ['tony', 'jack-sparrow', 'name', 'results']
    
#    for file in Path(logs).glob('*.obj'):
#        mesh = Mesh(file)
#        print(f'{file.as_posix():40s} {str(mesh.max.round(6)):30s} {mesh.min.round(6)}')
#    
#    for file in Path(templates).glob('*.obj'):
#        mesh = Mesh(file)
#        mean, max, min = mesh.scaled(2)
#        print(f'{file.as_posix():40s} {str(max.round(6)):30s} {min.round(6)}')
     
    
    for file in Path(templates).glob('*.obj'):
        print(file.as_posix())
        
        mesh = Mesh(file)
        verts = mesh.verts
        verts = normalize(verts)
        
        plot_2D(verts)
   
    for file in obj_list:
        file = Path(logs) / (file + '.obj')
        
        print(file.as_posix())
        mesh = Mesh(file)
        verts = mesh.verts
        
        plot_2D(verts)

        
    
    
    
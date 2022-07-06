#Imports
from cv2 import transform
from my_package.model import InstanceSegmentationModel
from my_package.data.dataset import Dataset
from my_package.analysis.visualize import plot_visualization
from my_package.data.transforms.blur import  BlurImage
from my_package.data.transforms.crop import  CropImage
from my_package.data.transforms.flip import  FlipImage
from my_package.data.transforms.rescale import  RescaleImage
from my_package.data.transforms.rotate import  RotateImage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def experiment(annotation_file, segmentor, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        segmentor: The image segmentor
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''
    
    
    #Create the instance of the dataset.
    dataset=Dataset(annotation_file,[])

    imgs=[]
    #Iterate over all data items.
    for i in range(dataset.__len__()):
        item = dataset.__getitem__(i)
        image=item['image']
        imgs.append(image)
  
    #Get the predictions from the segmentor.
    segts=[]
    for i in range(dataset.__len__()):
        segt=segmentor.__call__(imgs[i])
        segts.append(segt)

    #Draw the segmentation maps on the image and save them.
    plot_visualization(imgs,segts,outputs,'1')
    
    #Do the required analysis experiments.
    
    rows=1
    cols=2
    for i in transforms:
        data=Dataset(annotation_file,[i])
        length=data.__len__()#reading annotaion file
        img=data.__getitem__(3)['image']
        segt=segmentor.__call__(img)
        plot_visualization([img],[segt],outputs,i.get_name())
        
        #plotting and saving on matplotlib
        img=plt.imread(outputs+'/'+i.get_name()+'.jpg')
        fig = plt.figure(figsize=(10,10))
        fig.add_subplot(rows,cols,1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(outputs+'/'+i.get_name())

        img=plt.imread('data/imgs/3.jpg')
        fig.add_subplot(rows,cols,2)
        plt.imshow(img)
        plt.axis('off')
        plt.title('data/imgs/3')
        plt.savefig('data/output/fig_'+i.get_name()+'.png')
        plt.show()
        '''
    img=plt.imread('data/output/Rescale_ratio_2.jpg')
    fig.add_subplot(rows,cols,3)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Rescale_ratio_2')

    img=plt.imread('data/output/Rescale_ratio_0.5.jpg')
    fig.add_subplot(rows,cols,4)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Rescale_ratio_0.5')

    img=plt.imread('data/output/Rotate_degrees_-45.jpg')
    fig.add_subplot(rows,cols,5)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Rotate_degree_-45')

    img=plt.imread('data/output/Rotate_degrees_90.jpg')
    fig.add_subplot(rows,cols,6)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Rotate_degree_90')
    '''

def main():
    segmentor = InstanceSegmentationModel()
    #creating objects for transforms
    transforms=[]
    # transforms.append(CropImage( ( , ) )
    transforms.append(FlipImage('horizontal'))
    transforms.append(BlurImage(3))
    transforms.append(RescaleImage(2))
    transforms.append(RescaleImage(0.5))
    transforms.append(RotateImage(90))
    transforms.append(RotateImage(-45))


    experiment('./data/annotations.jsonl', segmentor, transforms=transforms, outputs='data/output') # Sample arguments to call experiment()


if __name__ == '__main__':
    main()

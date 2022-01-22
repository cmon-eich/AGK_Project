#The whole class by now is only for manipulation of images of the png-format
import os
import csv
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from random import randrange

#only resizes pngs found in the given folder and its subfolders, other files are ignored
def resize(input_png_path='images/raw256x256/',output_png_path='images/raw128x128/',subfolders=['kreis/','kreuz/','unentschieden/'],new_size=(128,128)):
    for subfolder in subfolders:
        Path(output_png_path+subfolder).mkdir(parents=True, exist_ok=True)
        png_filenames = os.listdir(input_png_path+subfolder)
        for png_filename in png_filenames:
            _, extention = os.path.splitext(png_filename)
            if (extention == '.png'):
                png = Image.open(input_png_path+subfolder+png_filename).resize(new_size)
                png.save(output_png_path+subfolder+png_filename)

#rotates every png once 90, 180 and 270 degrees
def rotate(png_path='images/raw128x128/',subfolders=['kreis/','kreuz/','unentschieden/']):
    for subfolder in subfolders:
        png_filenames = os.listdir(png_path+subfolder)
        for png_filename in png_filenames:
            filename, extention = os.path.splitext(png_filename)
            if (extention == '.png'):
                for degree in [90,180,270]:
                    png = Image.open(png_path+subfolder+png_filename).rotate(degree)
                    png.save(png_path+subfolder+filename+'r'+str(degree)+'deg'+extention)

def grayen(png_path='images/raw128x128/',subfolders=['kreis/','kreuz/','unentschieden/']):
    for subfolder in subfolders:
        png_filenames = os.listdir(png_path+subfolder)
        for png_filename in png_filenames:
            _, extention = os.path.splitext(png_filename)
            if (extention == '.png'):
                png = Image.open(png_path+subfolder+png_filename).convert('L')
                png.save(png_path+subfolder+png_filename)

#turns the images (by now only png are affected) in the subfolders of the given input-path to grayscale-images with the given size.
#parameters are self-explanatory
#parameter nomClass=True means Class is saved as string. If False Class is saved as numeric (0 = draw, 1 = X, 2 = O)
#TODO the images have to be turned into (csv-)data. 
def from_img_to_data(input_png_path='images/raw256x256/',output_png_path='images/raw128x128/',subfolders=['kreis/','kreuz/','unentschieden/'],new_size=(128,128), createTestData=True, createHeader=True, nomClass=True, csv_fileaddition=''):
    csv_file_train = open('tictactoe'+str(new_size[0])+'x'+str(new_size[1])+csv_fileaddition+'_train.csv', 'w', newline='')
    writer_train = csv.writer(csv_file_train, delimiter=',')
    if createTestData:
        csv_file_test = open('tictactoe'+str(new_size[0])+'x'+str(new_size[1])+csv_fileaddition+'_test.csv', 'w', newline='')
        writer_test = csv.writer(csv_file_test, delimiter=',')
    if createHeader:
        #add header
        num_cols = new_size[0]*new_size[1]
        i = 0
        header_list = []
        while i < num_cols:
            i = i+1
            header_list.append('c'+str(i))
        header_list.append('class')
        writer_train.writerow(header_list)
        if createTestData:
            writer_test.writerow(header_list)
    for subfolder in subfolders:
        nom_val = (0,'none')[nomClass]
        if subfolder == 'kreuz/':
            nom_val = (1,'cross')[nomClass]
        if subfolder == 'kreis/':
            nom_val = (2,'circ')[nomClass]
        Path(output_png_path+subfolder).mkdir(parents=True, exist_ok=True)
        png_filenames = os.listdir(input_png_path+subfolder)
        for png_filename in png_filenames:
            filename, extention = os.path.splitext(png_filename)
            if (extention == '.png'):
                png = Image.open(input_png_path+subfolder+png_filename).resize(new_size).convert('L')
                png.save(output_png_path+subfolder+png_filename)
                png_data = np.array(png).flatten().tolist()
                png_data.append(nom_val)
                writer = writer_train
                if randrange(100) > 95:
                    writer = writer_test
                writer.writerow(png_data)                    
                flip_and_mirror(png, writer, output_png_path+subfolder, filename, extention, nom_val)
                #no need to close png as PIL does it itself
                for degree in [90,180,270]:
                    png = png.rotate(degree)
                    png.save(output_png_path+subfolder+filename+'r'+str(degree)+'deg'+extention)
                    png_data = np.array(png).flatten().tolist()
                    png_data.append(nom_val)
                    writer.writerow(png_data)
                    flip_and_mirror(png, writer, output_png_path+subfolder, filename+'r'+str(degree)+'deg', extention, nom_val)
    csv_file_test.close()
    csv_file_train.close()

def flip_and_mirror(png, writer, output_png_path, filename, extention, nom_val):
    png_flip = ImageOps.flip(png)
    png_flip.save(output_png_path+filename+'_flipped'+extention)
    png_data = np.array(png_flip).flatten().tolist()
    png_data.append(nom_val)
    writer.writerow(png_data)
    png_mirror = ImageOps.mirror(png)
    png_mirror.save(output_png_path+filename+'_mirrored'+extention)
    png_data = np.array(png_mirror).flatten().tolist()
    png_data.append(nom_val)
    writer.writerow(png_data)

from_img_to_data(input_png_path='images/unified_V2.0_256x256/', output_png_path='images/uni64x64/', new_size=(64,64), createHeader=False, nomClass=False, csv_fileaddition='_V2')

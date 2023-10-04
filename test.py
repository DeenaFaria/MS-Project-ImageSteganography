import time
import math

from PIL import Image
from cv2 import imread,imwrite
import zlib

import numpy as np
from numpy import array
from base64 import urlsafe_b64encode
from hashlib import md5
from cryptography.fernet import Fernet
from custom_exceptions import *
from statistics import mean
from math import log10, sqrt
import cv2

def con(im):
    with open(im, 'rb') as f:
         image_data = f.read()

# Compress image data
    compressed_data = zlib.compress(image_data)

# Write compressed data to file
    with open('compressed_image.zlib', 'wb') as f:
         f.write(compressed_data)

    return 'compressed_image.zlib'

def decom(im):
    with open('compressed_image.zlib', 'rb') as f:
         compressed_data = f.read()

# Decompress data
    image_data = zlib.decompress(compressed_data)

# Write decompressed data to file
    with open('decompressed_image.png', 'wb') as f:
         f.write(image_data)

    return 'decompressed_image.png'

def avg(stego):
    asc=[]
    sss=stego
    for i in sss:
       aaa=ord(i)
    asc.append(aaa)

    #print (mean(asc))

    return mean(asc)

def rgb_to_grayscale(r, g, b):
    grayscale_value = 0.299 * r + 0.587 * g + 0.114 * b
    return int(grayscale_value)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

#Returns binary representation of a string
def str2bin(string):
    return ''.join((bin(ord(i))[2:]).zfill(7) for i in string)

#Returns text representation of a binary string
def bin2str(string):
    return ''.join(chr(int(string[i:i+7],2)) for i in range(len(string))[::7])

#Returns the encrypted/decrypted form of string depending upon mode input
def encrypt_decrypt(string,password,mode='enc'):
    _hash = md5(password.encode()).hexdigest()
    cipher_key = urlsafe_b64encode(_hash.encode())
    cipher = Fernet(cipher_key)
    if mode == 'enc':
        return cipher.encrypt(string.encode()).decode()
    else:
        return cipher.decrypt(string.encode()).decode()


#Encodes secret data in image
def encode(input_filepath,text,output_filepath,avg,password=None,progressBar=None):
    if password != None:
        data = encrypt_decrypt(text,password,'enc') #If password is provided, encrypt the data with given password
    else:
        data = text
    lm=len(data) #length of the message
    data_length = bin(len(data))[2:].zfill(32)
    

    bin_data = iter(data_length + str2bin(data))
   
    img = imread(input_filepath,1)
    if img is None:
        raise FileError("The image file '{}' is inaccessible".format(input_filepath))
    height,width = img.shape[0],img.shape[1]
    encoding_capacity = height*width*3
    total_bits = 32+len(data)*7
    if total_bits > encoding_capacity:
        raise DataError("The data size is too big to fit in this image!")
    completed = False
    modified_bits = 0
    progress = 0
    progress_fraction = 1/total_bits
    
    k1=avg #user provided stego key
    lp=encoding_capacity-k1 

    if avg<=(encoding_capacity-lm):
        k2=(encoding_capacity-avg)%lm #define a new stego key
        i=k2
        j=k2
        ini=k2 #seed point
        print("Your Stego Key: "+str(k2))
    else:
        i=k1
        j=k1
        ini=k1 #seed point
        print("Your Stego Key: "+str(k1))

    #lp=encoding_capacity-k1
    
    embedding_rounds = int(math.sqrt(lp - lm))
    
    embedding_counter = 0
    print("Embedding Round is "+str(embedding_rounds))

    start_time = time.time()  # Record the start time
    last_update_time = start_time
    update_interval = 0.5  # Update progress every 0.5 seconds

    for _ in range(embedding_rounds):
        
        for i in range(height):

            for j in range(width):
                pixel = img[i,j]
                r, g, b = pixel
                gray_value = rgb_to_grayscale(r, g, b)
                bit = gray_value  % 2
                if bit == 0:
                   i += 1  # Move to the next pixel
                   if i >= height:  # Ensure i stays within bounds
                        i = 0
                
                pixel = img[i,j] #Update pixel value
                for k in range(3):
                    try:
                        x = next(bin_data)
                    except StopIteration:
                        completed = True
                        break
                    if x == '0' and pixel[k]%2==1:
                        pixel[k] -= 1
                        modified_bits += 1
                    elif x=='1' and pixel[k]%2==0:
                        pixel[k] += 1
                        modified_bits += 1
                    if progressBar != None: #If progress bar object is passed
               
                        progressBar.setValue(progress*100)
                if completed:
                    break
            if completed:
                break

        embedding_counter += 1
        progress = embedding_counter / embedding_rounds
        
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            elapsed_time = current_time - start_time
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            print("\n")
            print(f"Encoding progress: {progress:.2%} | Remaining time: {remaining_time:.2f} seconds", end="\r")
            last_update_time = current_time
    
    print("\nEncoding completed.")        

    written = imwrite(output_filepath,img)
    if not written:
        raise FileError("Failed to write image '{}'".format(output_filepath))
    loss_percentage = (modified_bits/encoding_capacity)*100
    return written

#Extracts secret data from input image
def decode(em,avg,input_filepath,password=None,progressBar=None):
    result,extracted_bits,completed,number_of_bits = '',0,False,None
    

    img = imread(input_filepath)
    if img is None:
        raise FileError("The image file '{}' is inaccessible".format(input_filepath))
    height,width = img.shape[0],img.shape[1]
    i, j, ini = avg, avg, avg
    
    embedding_rounds = em
    embedding_counter = 0

    for _ in range(embedding_rounds):
        
        for i in range(height):
           
            for j in range(width):
                
                r, g, b = img[i,j]
                gray_value = rgb_to_grayscale(r, g, b)
                bit = gray_value  % 2
                if bit == 0:
                   i += 1  # Move to the next pixel
                   if i >= height:  # Ensure i stays within bounds
                        i = 0
             
                for k in img[i,j]:
                    result += str(k%2)
                    extracted_bits += 1
                    if progressBar != None and number_of_bits != None: #If progress bar object is passed
                        progressBar.setValue(100*(extracted_bits/number_of_bits))
                    if extracted_bits == 32 and number_of_bits == None: #If the first 32 bits are extracted, it is our data size. Now extract the original data
                        number_of_bits = int(result,2)*7
                        result = ''
                        extracted_bits = 0
                    elif extracted_bits == number_of_bits:
                        completed = True
                        break
                if completed:
                    break
            if completed:
                break
        if password == None:
            return bin2str(result)
        else:
            try:
                return encrypt_decrypt(bin2str(result),password,'dec')
            except:
                raise PasswordError("Invalid password!")
if __name__ == "__main__":

    

    ch = int(input('What do you want to do?\n\n1.Encrypt\n2.Decrypt\n\nInput(1/2): '))
    if ch == 1:
        ip_file = input('\nEnter cover image name(path)(with extension): ')
       
        text = input('Enter secret data: ')
        pwd = input('Enter Password: ')
        stego = input('Enter Stego key: ')
        aveg=avg(stego)
        op_file = input('Enter output image name(path)(with extension): ')

        
        try:
            original = cv2.imread(ip_file)

        
            Encoded = encode(ip_file,text,op_file,aveg,pwd)
            print(Encoded)
            
            compressed = con(op_file)

        except FileError as fe:
            print("Error: {}".format(fe))
        except DataError as de:
            print("Error: {}".format(de))
        else:
   
            print("Encoded Successfully!\n")
            value = PSNR(original, Encoded)
            print(f"PSNR value is {value} dB")
            Encoded = cv2.imread(op_file)
            cv2.imshow("original image", original)
            cv2.imshow("Encoded image", Encoded)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
       
    elif ch == 2:
        ip_file = input('Enter image path: ')
        ip_file=decom(ip_file)
        pwd = input('Enter password: ')
        stego = int(input('Enter Stego key: '))
        em = int(input('Enter the number of embedding rounds (if any): '))

        try:
            data = decode(em,stego,ip_file,pwd)
        except FileError as fe:
            print("Error: {}".format(fe))
        except PasswordError as pe:
            print('Error: {}'.format(pe))
        else:
            print('Decrypted data:',data)
    else:
        print('Wrong Choice!')

    

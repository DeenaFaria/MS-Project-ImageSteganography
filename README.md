Steganography Project

Description:
This project demonstrates a simple steganography implementation, allowing you to hide secret text within an image using LSB (Least Significant Bit) encoding. It provides functionalities to both encode and decode hidden messages within image files.

Features
Encryption: Securely encrypt your secret message before embedding it in the image using a password.
Decryption: Extract the hidden message from the encoded image by providing the correct password.
Stego Key: Choose a stego key that determines the initial pixel position for embedding data.
Progress Bar: Visualize the progress of encoding and decoding operations using a progress bar.

Installation
Clone or download the repository to your local machine.
Install the required Python packages by running: pip install -r requirements.txt

Usage
Run the Algo2.py script using Python.
Choose the operation you want to perform:
Encrypt: Encode a secret message into an image.
Decrypt: Extract a hidden message from an encoded image.

Encryption
Provide the input cover image file path.
Enter the secret data you want to hide.
Choose a password for encryption (optional).
Specify a stego key (a string that affects the embedding process).
Provide the output image file path for the encoded image.

Decryption
Provide the input encoded image file path.
Enter the password used during encryption (if applicable).
Specify the stego key used during encryption.
If you know the number of embedding rounds, enter it (optional).

Examples
Encryption

What do you want to do?

1.Encrypt
2.Decrypt

Input(1/2): 1

Enter cover image name(path)(with extension): cover_image.png
Enter secret data: This is a secret message!
Enter Password: mypassword
Enter Stego key: mystegokey
Enter output image name(path)(with extension): encoded_image.png
Your Stego Key: 117
Embedding Round is 4607417
Encoded Successfully!
PSNR value is 32.92759548852635 dB

Decryption

What do you want to do?

1.Encrypt
2.Decrypt

Input(1/2): 2

Enter image path: encoded_image.png
Enter password: mypassword
Enter Stego key: 117
Enter the number of embedding rounds (if any): 4607417
Decrypted data: This is a secret message!


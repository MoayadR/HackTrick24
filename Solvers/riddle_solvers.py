# Add the necessary imports here
import numpy as np
import pandas as pd
import spacy
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import inflect
from statsmodels.tsa.arima.model import ARIMA
from collections import deque
from SteganoGAN.utils import decode
from PIL import Image
import pickle

medium_model = pickle.load(open('ml_medium.pkl', 'rb'))
object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()

# def solve_cv_easy(test_case: tuple) -> list:
#     # shredded_image, shred_width = test_case
#     # shredded_image = np.array(shredded_image)
#     """
#     This function takes a tuple as input and returns a list as output.

#     Parameters:
#     input (tuple): A tuple containing two elements:
#         - A numpy array representing a shredded image.
#         - An integer representing the shred width in pixels.

#     Returns:
#     list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
#     """
#     pass


# def solve_cv_medium(test_case: tuple) -> list:
#     combined_image , patch_image = test_case
#     """
#     This function takes a tuple as input and returns a list as output.

#     Parameters:
#     input (tuple): A tuple containing two elements:
#         - A numpy array representing the RGB base image.
#         - A numpy array representing the RGB patch image.

#     Returns:
#     list: A list representing the real image.
#     """
#     gray= cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)


#     result= cv2.matchTemplate(gray, patch_image, cv2.TM_CCOEFF_NORMED)
#     minivan, max_val, min_loc, max_loc= cv2.minMaxLoc(result)

#     height, width= patch_image.shape[:2]

#     top_left= max_loc
#     bottom_right= (top_left[0] + width, top_left[1] + height)
#     cv2.rectangle(combined_image, top_left, bottom_right, (0,0,255),5)

#     cv2.imshow('Rainforest', combined_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = input
    image = np.array(image)
    image = Image.fromarray(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    # Load the spaCy NLP model
    nlp_model = spacy.load("en_core_web_sm")

    # COCO dataset classes
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Mapping COCO classes to IDs
    coco_class_to_id = {class_name: i + 1 for i, class_name in enumerate(coco_classes)}

    # Inflect engine for singularization
    inflect_engine = inflect.engine()

    def count_relevant_objects(question, image):
        # Define transformations to preprocess the image
        preprocess_transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ])

        # Preprocess the image
        processed_image = preprocess_transform(image).unsqueeze(0)

       

        # Perform object detection
        with torch.no_grad():
            predictions = object_detection_model(processed_image)

        # Extract the object of interest from the question
        question_doc = nlp_model(question)
        object_of_interest = None
        for token in question_doc:
            if token.pos_ == 'NOUN':
                object_of_interest = token.text.lower()
                break

        object_of_interest = inflect_engine.singular_noun(object_of_interest) or object_of_interest
        relevant_labels = [coco_class_to_id.get(object_of_interest.lower(), None) + 1]

        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        # Filter out irrelevant objects based on labels, confidence scores, and the object of interest
        min_score_threshold = 0.5  # Example: minimum confidence score threshold
        relevant_count = sum(
            1 for label, score in zip(labels, scores) if label in relevant_labels and score >= min_score_threshold)

        return relevant_count

    return count_relevant_objects(extracted_question, image)


    



def solve_ml_easy(data) -> list:
    df = pd.DataFrame(data)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """

    # Convert 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set 'timestamp' column as index
    df.set_index('timestamp', inplace=True)

    # Fit ARIMA model
    model = ARIMA(df['visits'], order=(5,1,0))
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=50)

    return forecast.tolist()


def solve_ml_medium(inp: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    inp = np.array(inp).reshape(1,-1)
    return int(medium_model.predict(inp)[0])



def solve_sec_medium(img) -> str:
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    tensor_image = torch.tensor(img)

    return decode(tensor_image)

# def solve_sec_hard(input:tuple)->str:
#     """
#     This function takes a tuple as input and returns a list a string.

#     Parameters:
#     input (tuple): A tuple containing two elements:
#         - A key 
#         - A Plain text.

#     Returns:
#     list:A string of ciphered text
#     """
#     pass

def solve_problem_solving_easy(input_data: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """

    data = input_data[0]
    key = input_data[1]

    solution = {}
    for st in data:
        if solution.get(st) is not None:
            solution[st] += 1
        else:
            solution[st] = 1

    def custom_sort(item):
        return (-item[1], item[0])


    sorted_data = sorted(solution.items(), key=custom_sort)
    
    output = [x[0] for x in sorted_data[:key]]

    return output


def solve_problem_solving_medium(text: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    stack = deque()
    i = len(text)-1
    while i >= 0 :
        word = ""
        num = ""
        if text[i] == '[':
            top = stack.pop()
            while top != ']':
                word += top
                top = stack.pop()
            stack.append(word)
        elif text[i].isdigit():
            while i >= 0 and text[i].isdigit():
                num += text[i]
                i -= 1
            num = num[::-1]
            temp = stack.pop()
            word = int(num) * temp
            stack.append(word)
            continue
        else:
            stack.append(text[i])
        i -= 1
    output = ""
    i = 0
    while i < len(stack):
        c = stack.pop()
        if c.isalpha():
            output += c
    return output


DP = [[-1 for _ in range(101)] for _ in range(101)]

def rec_solver(curr_x , curr_y , end_x , end_y):
    if curr_x >= end_x or curr_y >= end_y:
        return 0
    
    if curr_x == end_x -1 and curr_y == end_y-1:
        return 1
    
    if DP[curr_x][curr_y] != -1:
        return DP[curr_x][curr_y]
    
    DP[curr_x][curr_y] = rec_solver(curr_x +1 , curr_y , end_x , end_y) + rec_solver(curr_x , curr_y+1 , end_x , end_y)
    return DP[curr_x][curr_y]

def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    return rec_solver(0,0 , input[0] , input[1])

def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """

    # Python3 code for the above approach

    # Hexadecimal to binary conversion

    def hex2bin(s):
        mp = {'0': "0000",
              '1': "0001",
              '2': "0010",
              '3': "0011",
              '4': "0100",
              '5': "0101",
              '6': "0110",
              '7': "0111",
              '8': "1000",
              '9': "1001",
              'A': "1010",
              'B': "1011",
              'C': "1100",
              'D': "1101",
              'E': "1110",
              'F': "1111"}
        bin = ""
        for i in range(len(s)):
            bin = bin + mp[s[i]]
        return bin

    # Binary to hexadecimal conversion

    def bin2hex(s):
        mp = {"0000": '0',
              "0001": '1',
              "0010": '2',
              "0011": '3',
              "0100": '4',
              "0101": '5',
              "0110": '6',
              "0111": '7',
              "1000": '8',
              "1001": '9',
              "1010": 'A',
              "1011": 'B',
              "1100": 'C',
              "1101": 'D',
              "1110": 'E',
              "1111": 'F'}
        hex = ""
        for i in range(0, len(s), 4):
            ch = ""
            ch = ch + s[i]
            ch = ch + s[i + 1]
            ch = ch + s[i + 2]
            ch = ch + s[i + 3]
            hex = hex + mp[ch]

        return hex

    # Binary to decimal conversion

    def bin2dec(binary):

        binary1 = binary
        decimal, i, n = 0, 0, 0
        while (binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary // 10
            i += 1
        return decimal

    # Decimal to binary conversion

    def dec2bin(num):
        res = bin(num).replace("0b", "")
        if (len(res) % 4 != 0):
            div = len(res) / 4
            div = int(div)
            counter = (4 * (div + 1)) - len(res)
            for i in range(0, counter):
                res = '0' + res
        return res

    # Permute function to rearrange the bits

    def permute(k, arr, n):
        permutation = ""
        for i in range(0, n):
            permutation = permutation + k[arr[i] - 1]
        return permutation

    # shifting the bits towards left by nth shifts

    def shift_left(k, nth_shifts):
        s = ""
        for i in range(nth_shifts):
            for j in range(1, len(k)):
                s = s + k[j]
            s = s + k[0]
            k = s
            s = ""
        return k

    # calculating xow of two strings of binary number a and b

    def xor(a, b):
        ans = ""
        for i in range(len(a)):
            if a[i] == b[i]:
                ans = ans + "0"
            else:
                ans = ans + "1"
        return ans

    # Table of Position of 64 bits at initial level: Initial Permutation Table
    initial_perm = [58, 50, 42, 34, 26, 18, 10, 2,
                    60, 52, 44, 36, 28, 20, 12, 4,
                    62, 54, 46, 38, 30, 22, 14, 6,
                    64, 56, 48, 40, 32, 24, 16, 8,
                    57, 49, 41, 33, 25, 17, 9, 1,
                    59, 51, 43, 35, 27, 19, 11, 3,
                    61, 53, 45, 37, 29, 21, 13, 5,
                    63, 55, 47, 39, 31, 23, 15, 7]

    # Expansion D-box Table
    exp_d = [32, 1, 2, 3, 4, 5, 4, 5,
             6, 7, 8, 9, 8, 9, 10, 11,
             12, 13, 12, 13, 14, 15, 16, 17,
             16, 17, 18, 19, 20, 21, 20, 21,
             22, 23, 24, 25, 24, 25, 26, 27,
             28, 29, 28, 29, 30, 31, 32, 1]

    # Straight Permutation Table
    per = [16, 7, 20, 21,
           29, 12, 28, 17,
           1, 15, 23, 26,
           5, 18, 31, 10,
           2, 8, 24, 14,
           32, 27, 3, 9,
           19, 13, 30, 6,
           22, 11, 4, 25]

    # S-box Table
    sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
             [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
             [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
             [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

            [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
             [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
             [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
             [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

            [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
             [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
             [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
             [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

            [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
             [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
             [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
             [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

            [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
             [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
             [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
             [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

            [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
             [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
             [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
             [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

            [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
             [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
             [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
             [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

            [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
             [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
             [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
             [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

    # Final Permutation Table
    final_perm = [40, 8, 48, 16, 56, 24, 64, 32,
                  39, 7, 47, 15, 55, 23, 63, 31,
                  38, 6, 46, 14, 54, 22, 62, 30,
                  37, 5, 45, 13, 53, 21, 61, 29,
                  36, 4, 44, 12, 52, 20, 60, 28,
                  35, 3, 43, 11, 51, 19, 59, 27,
                  34, 2, 42, 10, 50, 18, 58, 26,
                  33, 1, 41, 9, 49, 17, 57, 25]

    def encrypt(pt, rkb, rk):
        pt = hex2bin(pt)

        # Initial Permutation
        pt = permute(pt, initial_perm, 64)
        # print("After initial permutation", bin2hex(pt))

        # Splitting
        left = pt[0:32]
        right = pt[32:64]
        for i in range(0, 16):
            # Expansion D-box: Expanding the 32 bits data into 48 bits
            right_expanded = permute(right, exp_d, 48)

            # XOR RoundKey[i] and right_expanded
            xor_x = xor(right_expanded, rkb[i])

            # S-boxex: substituting the value from s-box table by calculating row and column
            sbox_str = ""
            for j in range(0, 8):
                row = bin2dec(int(xor_x[j * 6] + xor_x[j * 6 + 5]))
                col = bin2dec(
                    int(xor_x[j * 6 + 1] + xor_x[j * 6 + 2] + xor_x[j * 6 + 3] + xor_x[j * 6 + 4]))
                val = sbox[j][row][col]
                sbox_str = sbox_str + dec2bin(val)

            # Straight D-box: After substituting rearranging the bits
            sbox_str = permute(sbox_str, per, 32)

            # XOR left and sbox_str
            result = xor(left, sbox_str)
            left = result

            # # Swapper
            if (i != 15):
                left, right = right, left
        # print("Round ", i + 1, " ", bin2hex(left),
        # 	" ", bin2hex(right), " ", rk[i])

        # Combination
        combine = left + right

        # Final permutation: final rearranging of bits to get cipher text
        cipher_text = permute(combine, final_perm, 64)
        return cipher_text

    key = input[0]
    pt = input[1]

    # Key generation
    # --hex to binary
    key = hex2bin(key)

    # --parity bit drop table
    keyp = [57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4]

    # getting 56 bit key from 64 bit using the parity bits
    key = permute(key, keyp, 56)

    # Number of bit shifts
    shift_table = [1, 1, 2, 2,
                   2, 2, 2, 2,
                   1, 2, 2, 2,
                   2, 2, 2, 1]

    # Key- Compression Table : Compression of key from 56 bits to 48 bits
    key_comp = [14, 17, 11, 24, 1, 5,
                3, 28, 15, 6, 21, 10,
                23, 19, 12, 4, 26, 8,
                16, 7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55,
                30, 40, 51, 45, 33, 48,
                44, 49, 39, 56, 34, 53,
                46, 42, 50, 36, 29, 32]

    # Splitting
    left = key[0:28]  # rkb for RoundKeys in binary
    right = key[28:56]  # rk for RoundKeys in hexadecimal

    rkb = []
    rk = []
    for i in range(0, 16):
        left = shift_left(left, shift_table[i])
        right = shift_left(right, shift_table[i])
        combine_str = left + right
        round_key = permute(combine_str, key_comp, 48)
        rkb.append(round_key)
        rk.append(bin2hex(round_key))

    cipher_text = bin2hex(encrypt(pt, rkb, rk))

    return cipher_text

riddle_solvers = {
    # 'cv_easy': solve_cv_easy,
    # 'cv_medium': solve_cv_medium,
    # 'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard,
    'sec_medium_stegano': solve_sec_medium,
}

# input_data = (["pharaoh","sphinx","pharaoh","pharaoh","nile", "sphinx","pyramid","pharaoh","sphinx","sphinx"] , 3)
# print(solve_problem_solving_easy(input_data))


# print(solve_problem_solving_medium("3[d1[e2[l]]]"))

# print(solve_problem_solving_hard((3 , 2)))
# riddle_solvers['problem_solving_hard'](3,2)

# cv read image


# print(img.shape)
# test = (img , 64)
# order = solve_cv_easy(test)

# print(order)

# # reordering the shreds to form the original image

# shred_width = 64
# shredded_image = img

# num_shreds = shredded_image.shape[1] // shred_width
# print(num_shreds)
# shredded_image = shredded_image[: , :shred_width*num_shreds]
# shredded_image = shredded_image[: , order]
# shredded_image = shredded_image.reshape(shredded_image.shape[0] , -1, 3)
# print(shredded_image.shape)
# cv2.imwrite('reordered.jpg' , shredded_image)
# print("Image saved successfully")


# img = Image.open('SteganoGAN/sample_example/encoded.png')
# img = np.array(img)

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# tensor_image = transform(img)

# tensor_image = tensor_image.unsqueeze(0)

# print(solve_sec_medium(tensor_image))



# combined__image = cv2.imread('../Riddles/cv_medium_example/combined_large_image.png')
# patch = cv2.imread('../Riddles/cv_medium_example/patch_image.png',0)
# # Example usage
# remove_and_interpolate('../Riddles/cv_medium_example/combined_large_image.png', '../Riddles/cv_medium_example/patch_image.png')

# print(solve_cv_medium((combined__image , patch)))

# print(solve_sec_hard(('133457799BBCDFF1' , '0123456789ABCDEF')))

# print(solve_problem_solving_medium("3[d1[e2[l]]]"))
# img = Image.open('../Riddles/cv_hard_example/cv_hard_sample_image.jpg')
# # conert to list
# img = np.array(img)
# print(solve_cv_hard(("How many cats are there??", img)))

# print(solve_sec_hard(('133457799BBCDFF1' , '0123456789ABCDEF')))
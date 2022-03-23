from PIL import Image
import cv2,numpy as np,math

class node:
    # Define node construction method
    def __init__(self, right=None, left=None, parent=None, weight=0, code=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.weight = weight  # weight
        self.code = code  # Node value

def picture_convert(filename, newfilename):
    picture = Image.open(filename)
    picture = picture.convert('L')  # Convert bmp picture to gray value picture
    picture.save(newfilename)  # Save grayscale image
    return picture

def pixel_number_caculate(list):
    pixel_number = {}
    for i in list:
        if i not in pixel_number.keys():
            pixel_number[i] = 1  # If the pixel is not in the character frequency dictionary, add it directly
        else:
            pixel_number[
                i] += 1  # If it exists in the character frequency string dictionary, add one to the corresponding value
    return pixel_number

def node_construct(pixel_number):
    node_list = []
    for i in range(len(pixel_number)):
        node_list.append(node(weight=pixel_number[i][1], code=str(pixel_number[i][0])))
    return node_list

def node_construct(pixel_number):
    node_list = []
    for i in range(len(pixel_number)):
        node_list.append(node(weight=pixel_number[i][1], code=str(pixel_number[i][0])))
    return node_list

def tree_construct(listnode):
    listnode = sorted(listnode, key=lambda node: node.weight)
    while len(listnode) != 1:
        # Each time the two pixel points of the weighted value are merged
        low_node0, low_node1 = listnode[0], listnode[1]
        new_change_node = node()
        new_change_node.weight = low_node0.weight + low_node1.weight
        new_change_node.left = low_node0
        new_change_node.right = low_node1
        low_node0.parent = new_change_node
        low_node1.parent = new_change_node
        listnode.remove(low_node0)
        listnode.remove(low_node1)
        listnode.append(new_change_node)
        listnode = sorted(listnode, key=lambda node: node.weight)
    return listnode

def Huffman_Coding(picture):
    width = picture.size[0]
    height = picture.size[1]
    im = picture.load()
    list = []
    for i in range(width):
        for j in range(height):
            list.append(im[i, j])
    pixel_number = pixel_number_caculate(list)
    pixel_number = sorted(pixel_number.items(), key=lambda item: item[1])

    node_list = node_construct(pixel_number)
    head = tree_construct(node_list)[0]
    coding_table = {}
    for e in node_list:
        new_change_node = e
        coding_table.setdefault(e.code, "")
        while new_change_node != head:
            if new_change_node.parent.left == new_change_node:
                coding_table[e.code] = "1" + coding_table[e.code]
            else:
                coding_table[e.code] = "0" + coding_table[e.code]
            new_change_node = new_change_node.parent
    coding_result = ''
    for i in range(width):
        for j in range(height):
            for key, values in coding_table.items():
                if str(im[i, j]) == key:
                    coding_result = coding_result + values
    file = open('coding_result.txt', 'w')
    file.write(coding_result)

def Decoding(width, height, coding_table, coding_result):
    code_read_now = ''
    new_pixel = []
    i = 0
    while (i != coding_result.__len__()):
        # Read one later each time
        code_read_now = code_read_now + coding_result[i]
        for key in coding_table.keys():
            # If the currently read code exists in the code table
            if code_read_now == coding_table[key]:
                new_pixel.append(key)
                code_read_now = ' '
                break
        i += 1
    decode_image = Image.new('L', (width, height))
    k = 0

    for i in range(width):
        for j in range(height):
            decode_image.putpixel((i, j), (int(new_pixel[k])))
        k += 1
    decode_image.save('decode.jpeg')

picture = picture_convert('strawberries.jpeg', 'huffman.jpeg')
Huffman_Coding(picture)
img = cv2.imread('strawberries.jpeg')
img2 = cv2.imread('huffman.jpeg', 1)

def PSNR(img, img2):
    mse = np.mean((img - img2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

print("PSNR value of Huffman method: ",PSNR(img,img2))
Y = np.square(np.subtract(img,img2)).mean()
print("MSE value of huffman method:",str(Y))

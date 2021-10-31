
# <Page imageFilename="0005-cr.jpg" imageWidth="5960" imageHeight="8983">
# <TextLine id="r1l2" custom="readingOrder {index:0;}">
#                <Coords points="3538,889 3569,887 3601,886 3632,885 3664,884 3695,883 3727,881 3758,881 3790,879 3821,877 3853,875 3884,874 3916,873 3947,872 3979,871 4010,871 4042,869 4073,869 4105,869 4136,869 4168,871 4168,790 4136,788 4105,788 4073,788 4042,788 4010,790 3979,790 3947,791 3916,792 3884,793 3853,794 3821,796 3790,798 3758,800 3727,800 3695,802 3664,803 3632,804 3601,805 3569,806 3538,808"/>
#                <Baseline points="3538,862 3569,860 3601,859 3632,858 3664,857 3695,856 3727,854 3758,854 3790,852 3821,850 3853,848 3884,847 3916,846 3947,845 3979,844 4010,844 4042,842 4073,842 4105,842 4136,842 4168,844"/>
#                <TextEquiv>
#                    <Unicode>Толковые</Unicode>
#                </TextEquiv>
#            </TextLine>


from PIL import Image
import xml.dom.minidom
import string
import glob
import imageio
import os
import math
from PIL import Image

def parseReadingOrder(str):
    start = '{index:'
    end = ';}'
    res = (str[str.find(start)+len(start):str.find(end)])
    return res

def parseXML(xmlfile):
    save_flag = 1
    lines_lst = []
    targets_descr = []
    doc = xml.dom.minidom.parse(xmlfile)
    pages = doc.getElementsByTagName("Page")
    for page in pages:
        imageFilename = page.getAttribute('imageFilename')
        imageWidth = page.getAttribute('imageWidth')
        imageHeight = page.getAttribute('imageHeight')


    lines = page.getElementsByTagName("TextLine")
    #print ("%d TextLine:" % lines.length)
    for line in lines:
        save_flag = 1
        readingOrder = parseReadingOrder(line.getAttribute("custom"))
        line_id = line.getAttribute("id")
        #print(line.getAttribute("id"), line.getAttribute("custom"))
        for node in line.getElementsByTagName('Coords'):
            points = node.getAttribute("points")
           # print(points)
            
        unicode_node = line.getElementsByTagName('Unicode')
        if unicode_node:
            for node in line.getElementsByTagName('Unicode'):
                try:
                    target = node.childNodes[0].nodeValue
                except: 
                    print("ERROR ", line_id)
                    save_flag = 0
                    continue
               # print(target)
        if(save_flag == 1 and unicode_node):
            line_descr = {'lineID': line_id, 'readingOrder': readingOrder, 'points': points, 'target': target}
            targets_descr.append({'lineID': line_id, 'readingOrder': readingOrder, 'target': target})     
            lines_lst.append(line_descr)
        
    saveTargets(imageFilename, targets_descr)
    saveTargetsForAttention(imageFilename, targets_descr)
    res = {'imageFilename': imageFilename, 'imageWidth': imageWidth, 'imageHeight': imageHeight, 'lines': lines_lst}
    
    return res


def cropLine(imgName, readingOrder, points):
    l_part = []
    r_part = []
    img = Image.open(imgName)
    size = len(imgName)
    #remove last 4 characters
    name = imgName[:size - 4]
    txt = "dataset_ru/outcome.txt"
    
    crdnts = points.split(",")
    crdnts = crdnts[1:-1]    
    for i in crdnts:
        l,r = i.split(" ")
        l_part.append(l)
        r_part.append(r)

    l_part = list(map(int, l_part))
    r_part = list(map(int, r_part))  
    left = int(min(r_part))
    top = int(min(l_part))
    right = int(max(r_part))
    bottom = int(max(l_part))
    
    cropped_img = img.crop((left, top, right, bottom))   
    
    start = 'images/'
    res1 = (name[name.find(start)+len(start):])
    res2 = (name[:name.find(start)+len(start)])
    res = res2 + 'lines/' + res1 

    cropped_name = res + "-" + readingOrder + ".png"
    cropped_name = cropped_name.replace(' ', '_')
    
    cropped_img.save(cropped_name)

    im = imageio.imread(cropped_name)
    #imageio.imwrite(cropped_name, im[:, :, 0])
    try:
        imageio.imwrite(cropped_name, im[:, :, 0])        
    except:
        imageio.imwrite(cropped_name, im)
    
    
    #res = (cropped_name[cropped_name.find(start)+len(start):])
    with open(txt, "a", encoding = "utf-8") as f:
        f.write(res1 + "-" + readingOrder + ".png" + "\n")
     
    return cropped_name + " was saved"


def saveTargets(imgName, targets_descr):
    txt = "dataset_ru/transcriptions.txt"
    size = len(imgName)
    name = imgName[:size - 4]   
    with open(txt, "a", encoding = "utf-8") as f:    
        for line in targets_descr:
            readingOrder = line.get('readingOrder')
            line_id = line.get('lineID')
            target = line.get('target')
            line_name = name + "-" + line_id + "-" + readingOrder 
            line_name = line_name.replace(' ', '_')
            split_line = splitToChars(splitToWords(target))
            full_line = line_name + " " + split_line + "\n"
            f.write(full_line)
            
    msg = imgName + " targets saved"
    return msg

def saveTargetsForAttention(imgName, targets_descr):
    txt = "C:/transcriptions.txt"
    size = len(imgName)
    name = imgName[:size - 4]   
    with open(txt, "a", encoding = "utf-8") as f:    
        for line in targets_descr:
            readingOrder = line.get('readingOrder')
            line_id = str(line.get('lineID'))
            target = line.get('target')
            line_name = name + "-" + line_id + "-" + readingOrder 
            #split_line = splitToChars(splitToWords(target))
            line_name = line_name.replace(' ', '_')
            full_line = line_name + ".png" + "|" + target + "\n"
            f.write(full_line)
            
    msg = imgName + " targets saved"
    return msg

def splitToWords(line):
    for s in string.punctuation:
        line = line.replace(s, " "+s)
    line = line.replace(" ", "|")
    return line



def splitToChars(line):
    new_line = ''
    for s in line:
        if(s =='|'):
            new_line = new_line[:-1] + s
        else:
            new_line = new_line + s +"-"
    return new_line[:-1]



def iterateImages(folder):
    for filepath in glob.iglob(folder+'/*.xml'):
        print(filepath)
        parsed = parseXML(filepath)
        imageFilename = parsed.get('imageFilename')
        imageFilename = folder + '/images/' + imageFilename
        print(imageFilename)
        lines = parsed.get('lines')
        for line in lines:
            readingOrder = line.get('readingOrder')
            line_id = line.get('lineID')
            points = line.get('points')
            cropLine(imageFilename, line_id+"-"+readingOrder, points)
    return folder


def saveSetsForAttentionOCR_lines(path, txt):
    import os
    import random
    
    train = path + "/train/transcriptions.txt" 
    test =  path + "/test/transcriptions.txt"
    valid = path + "/valid/transcriptions.txt"
    
    train_size = 11000
    test_size = 1000
    valid_size = 1000    
    cnt = 0
    

    with open(txt, encoding = "utf-8") as f:
        lines = f.readlines()
       
        for cnt in range(train_size+valid_size+test_size):        
            random_int = random.randint(0,len(lines)-1)
            line = lines[random_int]
  
            if cnt <= train_size:
                with open(train, "a", encoding = "utf-8") as train_file:    
                    train_file.write(line)
            
            if cnt > train_size and cnt <= train_size+valid_size:     
                with open(valid, "a", encoding = "utf-8") as valid_file:    
                    valid_file.write(line)
                    
            if cnt > train_size+valid_size and cnt <= train_size+valid_size+test_size:
                with open(test, "a", encoding = "utf-8") as test_file:    
                    test_file.write(line)
    print("done!")
    
    
fldr = 'C:/dataset_ru'
path = 'C:/Datasets/Russian'
transcriptions = 'C:/transcriptions.txt'

iterateImages(fldr) #parse xml in folder

saveSetsForAttentionOCR_lines(path, transcriptions) #split datasets


#filenames = os.listdir(path)
#half = 0.5
#
#for filename in filenames:    
#    name = os.path.join(path, filename)
#    img = Image.open(name)
#    w,h = img.size    
#    #if (w > 100 and h > 100): #1 step 
#    #if (w > 1000 or h > 1000): #2 step
#    if (w > 500 or h > 500): #2 step
#        print(filename)
#        img = img.resize( [int(half * s) for s in img.size] ,Image.ANTIALIAS )
#        img.save(name, quality=95)
#print("done!")
#    
#
#path =  'C:/line_images_normalized'
#filenames = os.listdir(path)
#
#for filename in filenames:
#    os.rename(os.path.join(path, filename), os.path.join(path, filename.replace(' ', '_')))
#
#print("renaming done!")

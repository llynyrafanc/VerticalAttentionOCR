
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
import pickle

def parseReadingOrder(str):
    start = '{index:'
    end = ';}'
    res = (str[str.find(start)+len(start):str.find(end)])
    return res

def parseXMLregions(xmlfile):
    save_flag = 1
    regions_lst = []
    targets_descr = []
    doc = xml.dom.minidom.parse(xmlfile)
    pages = doc.getElementsByTagName("Page")
    for page in pages:
        imageFilename = page.getAttribute('imageFilename')
        imageWidth = page.getAttribute('imageWidth')
        imageHeight = page.getAttribute('imageHeight')


    regions = page.getElementsByTagName("TextRegion")
    #print ("%d TextLine:" % lines.length)
    for region in regions:
        save_flag = 1 
        readingOrder = parseReadingOrder(region.getAttribute("custom"))
        region_id = region.getAttribute("id")
        #print(region.getAttribute("id"), region.getAttribute("custom"))
        for node in region.getElementsByTagName('Coords'):
            points = node.getAttribute("points")
            #print(points)
            break            
 
        unicode_node = region.getElementsByTagName('Unicode')
        #print('unicode_node', unicode_node)
        if unicode_node:
            for node in region.getElementsByTagName('Unicode'):
                try:
                    target = node.childNodes[0].nodeValue
                except:
                    print("ERROR ", region_id)
                    save_flag = 0
                    continue

           # print(target)
        if(save_flag == 1 and unicode_node):
            region_descr = {'regionID': region_id, 'readingOrder': readingOrder, 'points': points, 'target': target}
            targets_descr.append({'regionID': region_id, 'readingOrder': readingOrder, 'target': target})        
            regions_lst.append(region_descr)
        
    #saveTargets(imageFilename, targets_descr)
    #saveTargetsForAttention(imageFilename, targets_descr)
    saveTargetsForAttention_lst(imageFilename, targets_descr)
    res = {'imageFilename': imageFilename, 'imageWidth': imageWidth, 'imageHeight': imageHeight, 'regions': regions_lst}
    
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
    res = res2 + 'regions/' + res1 

    cropped_name = res + "-" + readingOrder + ".png"
    cropped_name = cropped_name.replace(' ', '_') 
    
    cropped_img.save(cropped_name)
    im = imageio.imread(cropped_name)   
    try:
        imageio.imwrite(cropped_name, im[:, :, 0])        
    except:
        imageio.imwrite(cropped_name, im)    
    
    #res = (cropped_name[cropped_name.find(start)+len(start):])
    with open(txt, "a", encoding = "utf-8") as f:
        f.write(res1 + "-" + readingOrder + ".png" + "\n")
     
    return cropped_name + " was saved"


def saveTargetsForAttention(imgName, targets_descr):
    txt = "C:/regions_transcriptions.txt"
    size = len(imgName)
    name = imgName[:size - 4]   
    with open(txt, "a", encoding = "utf-8") as f:    
        for line in targets_descr:
            readingOrder = line.get('readingOrder')
            line_id = str(line.get('regionID'))
            target = line.get('target')
            line_name = name + "-" + line_id + "-" + readingOrder 
            #split_line = splitToChars(splitToWords(target))
            line_name = line_name.replace(' ', '_')  
            full_line = line_name + ".png" + "|" + target + "\n"
            f.write(full_line)
            
    msg = imgName + " targets saved"
    return msg
    

def saveTargetsForAttention_lst(imgName, targets_descr):
    txt = "C:/regions_transcriptions_lst.pkl"
    size = len(imgName)
    name = imgName[:size - 4]
    
    with open(txt, "rb") as f:
        data = pickle.load(f)

        
    for line in targets_descr:
        readingOrder = line.get('readingOrder')
        line_id = str(line.get('regionID'))
        target = line.get('target')
        line_name = name + "-" + line_id + "-" + readingOrder 
        #split_line = splitToChars(splitToWords(target))
        line_name = line_name.replace(' ', '_')  
        #full_line = line_name + ".png" + "|" + target + "\n"
        entry = {"image": line_name + ".png", "target": target}        
        data.append(entry)   
        #print(data)
   
        with open(txt, "wb+") as f: 
            pickle.dump(data, f)
 
            
    msg = imgName + " targets saved"
    return msg
     
    
def iterateImages(folder):
    data = []
    txt = "C:/regions_transcriptions_lst.pkl"
    with open(txt, "ab+") as f:        
        pickle.dump(data, f)      
               
    for filepath in glob.iglob(folder+'/*.xml'):
        print(filepath) 
        parsed = parseXMLregions(filepath)

    return folder


fldr = 'C:/dataset_ru'
iterateImages(fldr)


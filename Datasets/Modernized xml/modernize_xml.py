import xml.dom.minidom
import glob
import os
import re
import shutil
from russpelling import *
import xml.etree.ElementTree as ET



def iterateXML(folder):
    dir = folder + '/modernized_xml/'
    for filepath in glob.iglob(folder+'/*.xml'):        
        print(filepath)
        filename = os.path.basename(filepath)
        newFilename = dir + filename 
        print("newFilename", newFilename)
        shutil.copyfile(filepath, newFilename)
        parsed = updateNodes(filepath)
        f = open(newFilename, 'w', encoding = 'utf-8')
        parsed.writexml(f, encoding = 'utf-8')
        f.close()

    return folder

def updateNodes(XMLfile):
    doc = xml.dom.minidom.parse(XMLfile)
    nodes = doc.getElementsByTagName("Unicode")
    for node in nodes:
        try:
            val = node.childNodes[0].nodeValue        
            new_val = modernizeNode(val)
      #  print(val, " :", new_val)
            node.childNodes[0].nodeValue = new_val
        except:
            continue    
    return doc

def modernizeNode(node):
    tokens = node.splitlines(keepends=True)
    end_token = "\r"
    new_str = ''
    for word in tokens:
        try:
            new_str = new_str + normalize(word) + ' '   
        except:
            new_str = new_str + word + ' '
        if word.count(end_token) > 0:
            new_str = new_str + end_token
    new_str = re.sub(r'\s([?.!;:(),"](?:\s|$))', r'\1', new_str)
    return new_str.strip()


folder = 'C:/dataset/validation'
iterateXML(folder)

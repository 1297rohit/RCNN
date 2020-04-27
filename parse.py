import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        doc = etree.parse(xml_file)
        count = doc.xpath("count(//object)")
        root = tree.getroot()
        with open(str(xml_file)[0:-4]+".csv","w+") as f:
            f.write(str(int(count)))
        for member in root.findall('object'):
            value = (
                     member[4][0].text,
                     member[4][1].text,
                     member[4][2].text,
                     member[4][3].text
                     )
            coord = " ".join(value)
            
            
        
            with open(str(xml_file)[0:-4]+".csv","a") as f:
                    
                    f.write("\n")
                    f.write(coord)
                    
            
            #xml_list.append(value)
    
    


def main():
    image_path ="annotations"
    xml_df = xml_to_csv(image_path)
    #xml_df.to_csv('1.csv', index=None)
    print('Successfully converted xml to csv.')


main()

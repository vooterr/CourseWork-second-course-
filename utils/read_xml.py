import xml.etree.ElementTree as ET
tree = ET.parse('data/classification/oxford_pet/annotations/xmls/american_bulldog_10.xml')
root = tree.getroot()
for child in root:
    print(f"Тег: {child.tag}, Атрибуты: {child.attrib}, Текст: {child.text}")
    for item in child.items():
        print(item)
print(root.tag, root.attrib, root.text)
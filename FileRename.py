import os

colorPath = r"C:\Users\9426224\Desktop\Dataset\color"
depthPath = r"C:\Users\9426224\Desktop\Dataset\depth"
txtPath = r"C:\Users\9426224\Desktop\Dataset\text"

colorList = os.listdir(colorPath)
depthList = os.listdir(depthPath)
txtList = os.listdir(txtPath)

number = 47
colorNumber = number
depthNumber = number
txtNumber = number


for color in colorList:
    newName = 'color' + str(colorNumber) + '.jpg'
    os.rename(os.path.join(colorPath, color), os.path.join(colorPath, newName))
    colorNumber = colorNumber + 1
    print(color, " has been removed and new name is:", newName)

for depth in depthList:
    newName = 'depth' + str(depthNumber) + '.jpg'
    os.rename(os.path.join(depthPath, depth), os.path.join(depthPath, newName))
    depthNumber = depthNumber + 1
    print(depth, " has been removed and new name is:", newName)

for txt in txtList:
    newName = 'text' + str(txtNumber) + '.txt'
    os.rename(os.path.join(txtPath, txt), os.path.join(txtPath, newName))
    txtNumber = txtNumber + 1
    print(txt, " has been removed and new name is:", newName)

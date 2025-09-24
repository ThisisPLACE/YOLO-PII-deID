import os

root = r"D:\PLACE - Zotac\Nigeria\Abuja Ground\Insta 360 Pro\final_dataset_logo\blurred"

txt_name = os.path.join(root,'_dir-lst.txt')
jpgs_name = os.path.join(root,'_jpg-lst.txt')

txts =[]
jpgs =[]
c=0
for path, subdirs, files in os.walk(root):
    for name in files:
        txts.append(list((path,subdirs,name)))
        jpgs.append((name.replace(".txt",".jpg")))


with open (txt_name,'w') as file:
    for item in txts:
        #print (*item, sep='\t')
        x = '\t'.join([str(x) for x in item])
        file.write(x+'\n')

with open (jpgs_name,'w') as file:
    for item in jpgs:
        #print (*item, sep='\t')
        x = ''.join([str(x) for x in item])
        file.write(x+'\n')

print ('listed '+str(len(jpgs))+" jpgs")
print ('listed '+str(len(txts))+" txts")
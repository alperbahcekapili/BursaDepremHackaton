from ultralyticsplus import YOLO, render_result
import numpy as np
import pickle
# load model
model = YOLO('mshamrai/yolov8n-visdrone')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image



# set image
image = '/home/mathematician/Downloads/target.jpg'

# perform inference
results = model.predict(image,conf=0.15)

# observe results


location_list=[]
class_list=[]

for item in results[0].boxes:
    myxy = item.xyxy.tolist()
    location_list.append(myxy)
    mycls = item.cls.tolist()
    
    #mycls.append(myxy)
    class_list.append(mycls)


flat_list = [item for sublist in location_list for item in sublist]

whole_list=[]

for element in flat_list:
    temp=[]
    deltax=element[2]-element[0]
    deltay=element[3]-element[1]
    node2 = [element[0]+deltax,element[1]]
    node4 = [element[0],element[1]+deltay]
    tuple1 = (element[0],element[1])
    tuple2 = tuple(node2)
    tuple3 = (element[2],element[3])
    tuple4 = tuple(node4)
    temp.append(tuple1)
    temp.append(tuple2)
    temp.append(tuple3)
    temp.append(tuple4)
    whole_list.append(temp)  
print(class_list)        



#   write pickled data to file
with open('loclist.pkl', 'wb') as f:
    pickle.dump(whole_list, f)
with open('classlist.pkl', 'wb') as ff:
    pickle.dump(class_list,ff)



#f.write("Locations: \n")
#mylocstr = ' '.join([str(item) for item in location_list])


#print(mylocstr)
# f.write(mylocstr)
# f.write("\nClasses: \n")
# myclsstr = ' '.join([str(item) for item in class_list])
# f.write(myclsstr)


#f.close
render = render_result(model=model, image=image, result=results[0])
render.show()
import os


def main(txt_folder):
    txt_folder_list = os.listdir(txt_folder)
    biggest = 0
    smallest = 447921
    with open('C:\\Users\\DELL\\Desktop\\txt_front.txt', 'w+') as new_txt:
        for num, file in enumerate(txt_folder_list):
            f = open(txt_folder + file)
            label = f.readline()
            label_list = label.split(' ')
            new_list = []

            image_name =  label_list[0]
            for ind, item in enumerate(label_list):
                if item == '1' and ind % 5 == 0:
                    w = float(label_list[ind - 2]) - float(label_list[ind - 4])
                    h = float(label_list[ind - 1]) - float(label_list[ind - 3])
                    area=w*h
                    if area>biggest:
                        biggest=area
                        temp_big_name=image_name
                    if area < smallest:
                        smallest=area
                        temp_small_name=image_name
                    if 10000<area<1300000 :
                        new_list.append(image_name)
                        if float(label_list[ind - 4]) < 0:
                            label_list[ind - 4] = '0'
                        new_list.extend(label_list[ind - 4:ind + 1])
                        new_list_str = ' '.join(new_list)
                        new_txt.write(new_list_str + '\n')
                        print('writing:', num, image_name)
                        new_list = []
                    else:
                        temp=0
                        temp=1
            f.close()

        new_txt.close()
        print(biggest,temp_big_name,smallest,temp_small_name)
        print('finished')


if __name__ == '__main__':
    main('D:\cabin_data\detection_data_all\labels_front\\')

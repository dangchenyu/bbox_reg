import os


def main(txt_folder):
    txt_folder_list = os.listdir(txt_folder)
    with open(txt_folder + 'txt_back.txt', 'w+') as new_txt:
        for num,file in enumerate(txt_folder_list):
            f = open(txt_folder + file)
            label = f.readline()
            label_list = label.split(' ')
            new_list = []
            image_name = label_list[0]
            for ind, item in enumerate(label_list):
                if item == '1':
                    new_list.append(image_name)
                    new_list.extend(label_list[ind - 4:ind + 1])
                    new_list_str = ' '.join(new_list)
                    new_txt.write(new_list_str+'\n')
                    print('writing:',num,image_name)
                    new_list = []
            f.close()

        new_txt.close()
        print('finished')


if __name__ == '__main__':
    main('D:\cabin_data\detection_data_all\labels_front')

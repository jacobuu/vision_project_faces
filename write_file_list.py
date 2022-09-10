import os
import sys

def write_landmarks_file(lmk, file_name, directory, dir_name):
    cnt = 0
    d = {}
    with open(lmk) as landmarks_file:
            for line in landmarks_file:
                cnt += 1
                if cnt == 1 :
                    continue
                key = line.split('\t')[0]
                value = line.split('(')[1].split(')')[0].replace(',', ' ')
                d[int(key)] = value
    landmarks_file.close()

    with open(file_name + '.txt', 'w') as lmk_f:
        for i in range(1, 51):
            lmk_f.write(d[i])
            lmk_f.write('\n')
    lmk_f.close()


def main(argv):
    if len(sys.argv) < 2:
        print("Please insert the faces3D directory!")
        return False
    
    directory = argv[1]
    print(f"Directory: {directory}")
    files = []

    for dir_name in os.listdir(directory):
        dir_path = directory + '/' + dir_name + '/'
        
        file_name = dir_path + dir_name
        obj = file_name + '.obj'
        jpg = file_name + '.jpg'
        mtl = file_name + '.mtl'
        lmk = dir_path + 'landmarks.txt'
        
        print(f"Processing: {file_name}")

        if not os.path.isfile(obj):
            print(obj, ' could not read')
            return False
        if not os.path.isfile(mtl):
            print(mtl, ' could not read')
            return False
        if not os.path.isfile(lmk):
            print(lmk, ' could not read')
            return False
        if len(os.listdir(dir_path)) <= 4 :
            if not os.path.isfile(jpg):
                print(jpg, ' could not read')
                return False
            files.append('faces3D/' + dir_name + '/' + dir_name)

        write_landmarks_file(lmk, file_name, directory, dir_name)
    
    with open('Data\FaceCNN\CUSTOM\CUSTOM_base_filelist_noproblems.txt', 'w') as f:
        for file in files:
            f.write(file)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    main(sys.argv)


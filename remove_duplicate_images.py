import json
import os

JSON_FILE_PATHS = ['similar-covid-images.json','similar-normal-images.json','similar-pneumonia-images.json']

def main():
    for file_path in JSON_FILE_PATHS:
        remove_duplicates(file_path)


def remove_duplicates(json_file_path):
    """
    This will remove all the duplicates from an image, given in the json file
    The json has to have the following format:
    {
        'image1.png':['image2.png','image3.png']
        'image2.png':['image4.png','image5.png']
        ..-
    }
    That means that image1 & image2 are very similar, and image1 & image3 are very similar.


    Args:
        json_file_path (string): oath to the json file
    """
    import json
    count = 0
    with open(json_file_path) as f:
        similarities = json.load(f) 
        for _ , duplicates in similarities.items():
            for duplicate in duplicates:
                try:
                    os.remove(duplicate)
                except:
                    print('failed to delete...')
                count = count + 1
    print(f"Removed {count} images.")

if __name__ == '__main__':
    main()
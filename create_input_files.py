from utils import create_input_files

if __name__ == '__main__':
	create_input_files(dataset='coco',karpathy_json_path='../cpation data/dataset_coco.json',image_folder='/media/ssd/caption data/',captions_per_image=5,min_word_freq=5,outputfolder='/media/ssd/caption data/',max_len=50)
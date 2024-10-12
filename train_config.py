"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""

DATA_PATHS = {
    
    'videomatte': {
        'train': './matting-data/VideoMatte240K_JPEG_SD/train',
        'valid': './matting-data/VideoMatte240K_JPEG_SD/valid',
    },
#    'videomatte': {
#        'train': './matting-data/VideoMatte240K_JPEG_HD/train',
#        'valid': './matting-data/VideoMatte240K_JPEG_HD/valid',
#    },
    'imagematte': {
        'train': './matting-data/ImageMatte/train',
        'valid': './matting-data/ImageMatte/valid',
    },
    'background_images': {
        'train': './matting-data/BG-20k/train',
        'valid': './matting-data/BG-20k/valid',
    },
    'background_videos': {
        'train': './matting-data/DVM/train',
        'valid': './matting-data/DVM/valid',
    },
    
    
    'coco_panoptic': {
        'imgdir': './matting-data/coco/train2017/',
        'anndir': './matting-data/coco/panoptic_train2017/',
        'annfile': './matting-data/coco/annotations/panoptic_train2017.json',
    },
    'spd': {
        'imgdir': './matting-data/SuperviselyPersonDataset/img',
        'segdir': './matting-data/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        'videodir': './matting-data/YouTubeVIS/train/JPEGImages',
        'annfile': './matting-data/YouTubeVIS/train/instances.json',
    }
    
}

# Generate ImageNet Decoder and Resie Noise Image

## Usage 1 --> Generate dataset on file system
    from imagenet_s_gen import ImageTransfer
    ImageTransfer(root_dir='/your/val/images/root/path', meta_file='/meta/val.txt',
                  save_root='/your/save/root/path', decoder_type='pil',
                  transform_type='val', resize_type='pil-bilinear').write_to_filesystem()

## Usage 2 --> Generate one image real time (recommend and has been integrated in the code )
    from imagenet_s_gen import ImageTransfer
    image_gen = ImageTransfer(root_dir='/your/val/images/root/path', meta_file='/meta/val.txt',
                save_root='', decoder_type='pil',
                transform_type='val', resize_type='pil-bilinear')
    # generate numpy image on index 0
    numpy_image, label = image_gen.getimage(0)


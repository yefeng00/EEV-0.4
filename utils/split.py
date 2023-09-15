import os

height = 1080
width = 1920
org_dir = r"/home/hddb/klin/E2E/data/UVG/org"
img_dir = r"/home/hddb/klin/E2E/data/UVG/images_pad"

ffmpeg = r"ffmpeg"
for yuvname in os.listdir(org_dir):
    short_name = yuvname.split('_')[0]
    img_save_dir = os.path.join(img_dir, short_name)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    info = '{bin} -y -pix_fmt yuv420p -s {width}x{height} -i {yuv} -vframes 120 {save_path}' \
        .format(bin = ffmpeg, width = width, height = height, yuv = os.path.join(org_dir, yuvname), save_path = os.path.join(img_save_dir, 'im%03d.png'))

    print(info)
    os.system(info)



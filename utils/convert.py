import os

height = 1080
width = 1920
crop_height = 1024
crop_width = 1920
org_dir = r'/gpfs/share/home/1801111388/data/ClassB/org'
crop_dir = r'/gpfs/share/home/1801111388/data/ClassB/crop'
img_dir = r'/gpfs/share/home/1801111388/data/ClassB/images'

ffmpeg = r'/gpfs/share/home/1801111388/ffmpeg/ffmpeg-git-20210425-amd64-static/ffmpeg'
if not os.path.exists(crop_dir):
    os.makedirs(crop_dir)
for yuvname in os.listdir(org_dir):
    crop_yuvname = yuvname.replace(str(height), str(crop_height)).replace(str(width), str(crop_width))
    short_name = yuvname.split('_')[0]
    img_save_dir = os.path.join(img_dir, short_name)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    info = '{bin} -pix_fmt yuv420p -s {width}x{height} -i {org_yuv} -vframes 100 -vf scale="{crop_width}:{crop_height}" {crop_yuv}' \
        .format(bin = ffmpeg, width = width, height = height, org_yuv = os.path.join(org_dir, yuvname), crop_width = crop_width, crop_height = crop_height, crop_yuv = os.path.join(crop_dir, crop_yuvname))

    print(info)
    os.system(info)

    info = '{bin} -y -pix_fmt yuv420p -s {crop_width}x{crop_height} -i {crop_yuv} -vframes 100 {save_path}' \
        .format(bin = ffmpeg, crop_width = crop_width, crop_height = crop_height, crop_yuv = os.path.join(crop_dir, crop_yuvname), save_path = os.path.join(img_save_dir, 'im%03d.png'))

    print(info)
    os.system(info)



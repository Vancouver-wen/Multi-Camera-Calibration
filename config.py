from easydict import EasyDict

config=EasyDict()

# Calibration Board Parameters
# Default YMAL file in "multical/example_boards/my_charuco_16x22.yaml"
board=EasyDict()
board.size=[16,22] # 宽、长 各有多少个方块
board.square_length=0.05 # 方块内部间距大小
board.marker_length=0.0375 # 方块二维码的大小
# board.size=[8,10] # 宽、长 各有多少个方块
# board.square_length=0.1 # 方块内部间距大小
# board.marker_length=0.075 # 方块二维码的大小
config.board=board
config.board_size= "A0"
config.board_pixel_pre_mm= "2"

# Focus Parameters
config.adjustFocus=True

# CalibrationParameters
config.recollect=False  # 是否重新采集 校准图片
config.auto=True  # 人工采集校准图片 or 程序自动采集
config.img_path="./img_collect/"

config.response_threshold=20 # 一个图片成为合格的校准图像，所需要具备的“可用角点数量”
config.max_available_num=25 # 需要采集的“可用”图片数量

# 这里默认使用 A0 大小的尺寸
config.horizontal_table=(1189,841,0) # 桌子(width,length,height),单位 mm; 用于统一所有cam的外参

# Camera Parameters
config.cam_num=1
config.resolution=(960,960)

# Imshow Parameters
config.imgOrder=(1,1)
config.imgScale=1

# Save Control
config.save=True
config.save_path="./sequence/"

# Redis Control
config.redis=False
config.redis_ip="localhost"
config.redis_numpy_key="TODO"
config.redis_jpeg_key="TODO"

# Image Format
config.img_numpy=False
config.img_jpeg=True





import os

Cfg = {
    "secret": r"crack recognition system based on deep learning models",
    "model": "UNet",
    "checkpoints_dir": "checkpoints",
    "port": 20131,
    "server_dir": "/crack/server/",
    "lock_dir": "/crack/lock/",
    "model_dir": "/crack/model/",
    "train_dir": "/crack/train/",
    "token_dir": "/crack/token/",
    "token_ttl": 600,

    "etcd_ip": "162.14.67.90",
    "etcd_port": 2379,
    "etcd_timeout": 3,

    "mysql_ip": "162.14.67.90",
    "mysql_port": 33060,
    "mysql_database": "crack",
    "mysql_user": "root",
    "mysql_password": "crack123456",

    "originVideoDir": os.path.join('tmp', os.path.join('video', 'origin')),
    "resultVideoDir": os.path.join('tmp', os.path.join('video', 'result')),
    "h264VideoDir": os.path.join('tmp', os.path.join('video', 'h264'))
}


def init_config():
    if os.path.exists(Cfg.get("originVideoDir")) is False:
        os.makedirs(Cfg.get("originVideoDir"))

    if os.path.exists(Cfg.get("originVideoDir")) is False:
        os.makedirs(Cfg.get("originVideoDir"))

    if os.path.exists(Cfg.get("h264VideoDir")) is False:
        os.makedirs(Cfg.get("h264VideoDir"))




# import ffmpeg
# stream = ffmpeg.input('1.mp4')
# stream = ffmpeg.output(stream, 'h264.mp4',  vcodec="libx264").global_args('-y')
# ffmpeg.run(stream)

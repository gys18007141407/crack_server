# -*- coding: utf-8 -*-
import io
import flask
from flask import jsonify
from flask.globals import request

import numpy
from datasets import *
from evaluations import *
from utils import *
import cv2
import ffmpeg

import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from PIL import Image
import hashlib
import authlib.jose
from authlib.jose import jwt


net, needH, needW = name2net(Cfg.get("model"), 3, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_file = open(time.strftime("%Y-%m-%d_%H-%M-%S")+'.log', 'w', encoding="utf-8")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=log_file)


def predict_img(net,
                needH,
                needW,
                full_img,
                device,
                threshold):
    net.eval()
    # 处理图片：单通道增维，放缩图片
    img = torch.from_numpy(CrackDataset.preprocess(full_img))

    H, W = img.size(1), img.size(2)
    if needH is None:
        needH = H
    if needW is None:
        needW = W

    # 整张图预测结果
    result = numpy.zeros(shape=(H, W), dtype=np.uint8)

    # 局部预测结果
    for i in range(0, H, needH):
        if i + needH > H:
            i = H - needH
        for j in range(0, W, needW):
            if j + needW > W:
                j = W - needW
            partImg = img[:, i:i+needH, j:j+needW]
            # 增加batch维度
            partImg = partImg.unsqueeze(0).to(device=device, dtype=torch.float32)
            # 模型预测
            with torch.no_grad():
                partPred = net(partImg)
                probs = torch.sigmoid(partPred).squeeze(0)
                full_mask = probs.squeeze().cpu().numpy()
                result[i:i+needH, j:j+needW] = np.where(full_mask > threshold + 1e-3, 255, 0).astype(dtype=np.uint8)
    return result


def predict_video(net,
                needH,
                needW,
                video,
                name,
                frameFreq,   # 预测结果的帧率。默认与原视频帧率一样
                device,
                threshold):
    net.eval()
    # 帧率
    FPS = int(round(video.get(cv2.CAP_PROP_FPS)))
    # 分辨率-宽度
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 分辨率-高度
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 总帧数
    Frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    result = cv2.VideoWriter(os.path.join(Cfg.get('resultVideoDir'), os.path.join(str(flask.g.id), name)), cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), frameFreq, (W, H), isColor=False)
    cnt = 0

    while True:
        ok, img = video.read()
        # 视频读完
        if not ok:
            break
        # 预测这一帧
        img = Image.fromarray(img)

        # 转RGB
        img = img.convert("RGB")

        img = predict_img(net=net, needH=needH, needW=needW, full_img=img, device=device, threshold=threshold)
        # 构建视频
        result.write(img)

        cnt += 1
    result.release()
    video.release()
    cv2.destroyAllWindows()
    # 返回base64编码后的视频
    stream = ffmpeg.input(os.path.join(Cfg.get('resultVideoDir'), os.path.join(str(flask.g.id), name)))
    stream = ffmpeg.output(stream, os.path.join(Cfg.get('h264VideoDir'), os.path.join(str(flask.g.id), name)),  vcodec="libx264").global_args('-y')
    ffmpeg.run(stream)
    encoded = video2base64(os.path.join(Cfg.get('h264VideoDir'), os.path.join(str(flask.g.id), name)))
    pred = "data:video/mp4;base64," + encoded.decode().strip()
    return jsonify(result=pred)


# ==================== [CONFIG] ====================

app = flask.Flask(__name__)


@app.after_request
def add_header(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,OPTIONS,DELETE'
    r.headers['Access-Control-Allow-Headers'] = '*'
    r.headers['Access-Control-Allow-Credentials'] = 'true'
    r.headers['Access-Control-Max-Age'] = '3600'
    return r


@app.before_request
def validate_token():
    if request.method == "OPTIONS":
        return "", 200
    if request.path != "/api/v1/login" and request.path != '/api/v1/register':
        token = request.headers.get('Token')
        if token == "null":
            return jsonify(msg="请携带token"), 401
        key = Cfg.get("secret")
        try:
            claims = jwt.decode(token, key)
        except authlib.jose.errors:
            return jsonify(msg="token错误"), 401

        # token有效
        v, meta = etcd.get(key=Cfg.get("token_dir") + token)
        if v is None:
            return jsonify(msg="token过期"), 401

        lease_id = int.from_bytes(v, byteorder='big')
        etcd.refresh_lease(lease_id=lease_id)
        flask.g.id = claims['id']


@app.route("/api/v1/login", methods=["POST"])
def login():
    name = flask.request.json.get("name")
    password = flask.request.json.get("password")
    hash_obj = hashlib.sha256()
    hash_obj.update(password.encode())
    password = hash_obj.hexdigest()
    session = session_factory()
    user = session.query(User.id).filter(User.name == name, User.password == password).first()
    if user is None:
        session.close()
        return jsonify(msg="登录失败！用户不存在或者密码错误"), 401
    perm = session.query(Permission.role).filter(Permission.id == user.id).first()
    if perm.role == PermissionEnum.Admin:
        perm = 'admin'
    else:
        perm = 'user'
    session.close()
    return jsonify(token=create_token(user.id), role=perm)


@app.route("/api/v1/register", methods=["POST"])
def register():
    name = flask.request.json.get("name")
    password = flask.request.json.get("password")
    hash_obj = hashlib.sha256()
    hash_obj.update(password.encode())
    password = hash_obj.hexdigest()
    role = flask.request.json.get('role')

    session = session_factory()
    cnt = session.query(User.id).filter(User.name == name).count()
    if cnt > 0:
        session.close()
        return jsonify(err="用户名已存在, 注册失败!"), 401

    if role == 'admin':
        authorized_name = flask.request.json.get("authorized_name")
        authorized_password = flask.request.json.get("authorized_password")
        permission_id = session.query(Permission.id).filter(Permission.role == PermissionEnum.Admin).first().id
        user = session.query(User.id).filter(User.name == authorized_name, User.password == authorized_password).first()
        if user is None:
            session.close()
            return jsonify(msg="登录失败！授权账户信息错误!"), 401
        if user.permission != permission_id:
            session.close()
            return jsonify(msg="登录失败！该账户没有权限!"), 401
    else:
        permission_id = session.query(Permission.id).filter(Permission.role == PermissionEnum.User).first().id
    user = User(name=name, password=password, permission=permission_id)
    session.add(user)
    session.commit()
    user = session.query(User.id).filter(User.name == name).first()
    if os.path.exists(os.path.join(Cfg.get('originVideoDir'), str(user.id))) is False:
        os.makedirs(os.path.join(Cfg.get('originVideoDir'), str(user.id)))
    if os.path.exists(os.path.join(Cfg.get('resultVideoDir'), str(user.id))) is False:
        os.makedirs(os.path.join(Cfg.get('resultVideoDir'), str(user.id)))
    if os.path.exists(os.path.join(Cfg.get('h264VideoDir'), str(user.id))) is False:
        os.makedirs(os.path.join(Cfg.get('h264VideoDir'), str(user.id)))
    session.close()
    return jsonify(msg="注册成功!"), 200


@app.route("/api/v1/identify", methods=["POST"])
def identify():
    # 文件
    files = flask.request.json.get("files")
    # 类型
    types = flask.request.json.get("types")
    # 名称
    name = flask.request.json.get("name")
    # 分割阈值
    threshold = flask.request.json.get("threshold")

    if not isinstance(files, str) or not isinstance(types, str) or not isinstance(name, str):
        return jsonify(err="参数错误!"), 401

    if is_training():
        return jsonify(err="当前服务器正在训练!"), 401

    # base64解码为字节流
    try:
        files = files.split("base64,")[1]
    except IndexError:
        return jsonify(err="错误的文件编码方式!"), 401
    else:
        files = base64.b64decode(files)

    # 图片还是视频
    if types == "image":
        files = Image.open(io.BytesIO(files))
        # 转RGB
        files = files.convert("RGB")
        # 调用预测
        result = predict_img(net=net, needH=needH, needW=needW, full_img=files, threshold=float(threshold), device=device)
        # 返回base64编码后的图片
        pred = Image.fromarray(result)
        buffer = io.BytesIO()
        pred.save(buffer, format="PNG")
        pred = base64.b64encode(buffer.getvalue())
        pred = "data:image/png;base64," + pred.decode()
        return jsonify(result=pred)

    elif types == "video":
        with open(os.path.join(Cfg.get('originVideoDir'), os.path.join(str(flask.g.id), name)), "wb") as f:
            f.write(files)
        video = cv2.VideoCapture(os.path.join(Cfg.get('originVideoDir'), os.path.join(str(flask.g.id), name)))
        # 调用预测
        return predict_video(net=net, needH=needH, needW=needW, video=video, name=name, frameFreq=int(video.get(cv2.CAP_PROP_FPS)), threshold=float(threshold), device=device)

    return jsonify(err="错误的文件类型!"), 400


@app.route("/api/v1/servers", methods=["GET"])
def servers():
    tuples = etcd.get_prefix(key_prefix=Cfg.get("server_dir"))
    result = {}
    for value, meta in tuples:
        model_ip = value.decode().split('/')
        if model_ip[0] not in result.keys():
            result[model_ip[0]] = []
        result[model_ip[0]].append(model_ip[1])
    return jsonify(result=result)


@app.route("/api/v1/datasets", methods=["GET"]) # 访问MySQL获取数据集
def get_all_datasets():
    result = []
    session = session_factory()
    for items in session.query(CrackDatasets).all():
        result.append({items.name: items.desc})
    session.close()
    return jsonify(result=result)


@app.route("/api/v1/datasetsImages", methods=["GET"]) # 访问MySQL获取图片
def datasets_images():
    datasets = flask.request.args.get("datasets")
    t = flask.request.args.get("type")
    from_id = flask.request.args.get("from")
    count = flask.request.args.get("count")
    result = []

    session = session_factory()
    datasets_id = session.query(CrackDatasets.id).filter(CrackDatasets.name == datasets).first().id
    if t is None:
        temp = session.query(CrackImages).filter(CrackImages.datasets == datasets_id).limit(count).offset(from_id).all()
    else:
        type_id = session.query(CrackImagesType.id).filter(CrackImagesType.desc == t).first().id
        temp = session.query(CrackImages).filter(CrackImages.datasets == datasets_id, CrackImages.type == type_id).limit(count).offset(from_id).all()
    for items in temp:
        result.append({'name': items.name, 'image': base64.standard_b64encode(bytearray(items.image)).decode(),
                       'label': base64.standard_b64encode(bytearray(items.label)).decode()})
    session.close()
    return jsonify(result=result, total=len(temp))


@app.route("/api/v1/create_dataset", methods=["POST"])
def create_dataset():
    # 名称
    name = flask.request.json.get("name")
    desc = flask.request.json.get("desc")
    session = session_factory()

    cnt = session.query(CrackDatasets.id).filter(CrackDatasets.name == name).count()
    if cnt > 0:
        session.close()
        return jsonify(err="数据集已存在, 创建失败!"), 401

    dataset = CrackDatasets(name=name, desc=desc)
    session.add(dataset)
    session.commit()
    session.close()
    return jsonify(msg="创建成功!"), 200


@app.route("/api/v1/delete_image", methods=["DELETE"])
def delete_image():
    # 名称
    img_name = flask.request.json.get("img_name")
    dataset_name = flask.request.json.get("dataset_name")
    session = session_factory()

    dataset = session.query(CrackDatasets.id).filter(CrackDatasets.name == dataset_name).first()
    if dataset is None:
        return jsonify(msg="未找到该数据集!"), 402

    img = session.query(CrackImages.id).filter(CrackImages.name == img_name, CrackImages.datasets == dataset.id).first()
    if img is None:
        return jsonify(msg="未找到该图片!"), 402

    session.delete(img)
    session.commit()
    session.close()
    return jsonify(msg="删除成功")


@app.route("/api/v1/add_image", methods=["POST"])
def add_image():
    # 名称
    img_name = flask.request.json.get("img_name")
    dataset_name = flask.request.json.get("dataset_name")
    image = flask.request.json.get("image")
    label = flask.request.json.get("label")
    img_type = flask.request.json.get("type")
    session = session_factory()

    dataset = session.query(CrackDatasets.id).filter(CrackDatasets.name == dataset_name).first()
    if dataset is None:
        return jsonify(msg="未找到该数据集!"), 402

    t = session.query(CrackImagesType.id).filter(CrackImagesType.desc == img_type).first()
    if t is None:
        return jsonify(msg="未找到该类型!"), 402

    if session.query(CrackImages.id).filter(CrackImages.name == img_name, CrackImages.datasets == dataset.id).count() > 0:
        return jsonify(msg="图片名称在该数据集里面已存在!"), 402
    base64.standard_b64encode(bytes).decode()
    img = CrackImages(name=img_name, image=base64.standard_b64decode(image.encode()), label=base64.standard_b64decode(label.encode()), datasets=dataset.id, type=t.id)
    session.add(img)
    session.commit()
    session.close()
    return jsonify(msg="添加成功")


@app.route("/api/v1/train", methods=["POST"]) # 数据集本地不存在 => 从MySQL下载
def train():
    if is_training():
        return jsonify(err="当前服务器正在训练!"), 401

    dataset = flask.request.json.get("dataset")
    batch_size = int(flask.request.json.get("batch_size"))
    lr = float(flask.request.json.get("lr"))
    epochs = int(flask.request.json.get("epoch"))
    valid_interval = int(flask.request.json.get("valid_interval"))
    save_name = flask.request.json.get("name") # 最终保存的模型的名称
    fine_tune = flask.request.json.get("fine_tune") # 判断是否是微调当前模型

    # TODO:检查参数合法（包括数据集是否可用等等）

    # 服务器卸载，主动移除IP
    offload_ip()

    def train_func():
        # 训练则重置当前net参数，微调则继续训练当前net
        global net
        if fine_tune != 'true':
            net, _, _ = name2net(Cfg.get('model'), 3, 1)

        dataset_dir = os.path.join(os.path.join('datasets', dataset), 'cracks_cropped')
        train_dataset = CrackDataset(os.path.join(dataset_dir, 'train_imgs'), os.path.join(dataset_dir, 'train_labels'))
        dev_dataset = CrackDataset(os.path.join(dataset_dir, 'validation_imgs'),
                                   os.path.join(dataset_dir, 'validation_labels'))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
        val_loader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

        # TODO:甚至优化器和损失函数都支持自定义
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.875)
        criterion = DiceLoss()

        for epoch in range(1, epochs + 1):
            etcd.put(Cfg.get('train_dir') + get_ip() + '/' + Cfg.get('model'), '{} {}'.format(epoch, epochs).encode())

            epoch_loss = 0.
            for idx, batch in enumerate(train_loader, 1):
                imgs = batch['image']
                labels = batch['label']

                if imgs.shape[1] != net.n_channels:
                    # TODO: 怎么通知用户数据集错误
                    continue
                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)

                pred = net(imgs)

                loss = criterion(pred, labels)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                # 避免梯度消失
                nn.utils.clip_grad_value_(net.parameters(), 0.01)
                optimizer.step()

            if (epoch - 1) % valid_interval == 0:
                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
            torch.save(net.state_dict(), os.path.join(Cfg.get('checkpoints_dir'), os.path.join(Cfg.get('model'), save_name)))

        # 训练完成
        etcd.delete_prefix(prefix=Cfg.get('train_dir') + get_ip())

    # etcd注入训练信息，方便用户查询训练状态
    etcd.put(Cfg.get('train_dir') + get_ip(), get_ip().encode())
    # 起线程完成训练，不阻塞用户请求
    threading.Thread(target=train_func).start()

    # 通过/api/v1/train_list接口向用户暴露训练状态
    return jsonify(msg="已经开始训练!")


@app.route("/api/v1/train_list", methods=["GET"])
def train_list():
    tuples = etcd.get_prefix(key_prefix=Cfg.get("train_dir"))
    result = {}
    for value, meta in tuples:
        ip_model = meta.kv.key.removeprefix(Cfg.get('train_dir'))
        ip = ip_model.split('/')[0]
        model = ip_model.split('/')[1]
        epoch_epochs = value.decode().split(' ')
        epoch = epoch_epochs[0]
        epochs = epoch_epochs[1]
        result[ip].append([model, epoch, epochs])
    return jsonify(result=result)


@app.route("/api/v1/train_results", methods=["GET"])
def train_results():
    model_dir = os.path.join(Cfg.get('checkpoints_dir'), os.path.join(Cfg.get('model')))
    result = {}
    for item in os.listdir(model_dir):
        if os.path.isdir(item) or item == 'best_model':
            continue
        result[Cfg.get('model')].append(item)
    return jsonify(result=result)


if __name__ == "__main__":
    init_mysql_tables()

    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(Cfg.get('checkpoints_dir'), os.path.join(Cfg.get('model'), 'best_model')), map_location=device))

    app.run(host="0.0.0.0", port=20131, debug=True)

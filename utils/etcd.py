import etcd3
import sys
from .config import Cfg
from authlib.jose import jwt
import socket
import time
import threading


def connect_etcd(allow_fail=False):
    try:
        new_etcd_client = etcd3.client(host=Cfg.get("etcd_ip"),
                                       port=Cfg.get("etcd_port"),
                                       timeout=Cfg.get("etcd_timeout"),
                                       grpc_options={
                                           'grpc.max_connection_idle_ms': 2**31 - 1,
                                           'grpc.max_connection_age_ms': 2**31 - 1,
                                       }.items())
        new_etcd_client.get(key="/")
    except etcd3.exceptions.Etcd3Exception:
        if allow_fail:
            return None
        else:
            print("ETCD连接失败!")
            sys.exit(0)
    return new_etcd_client


etcd = connect_etcd()


def create_token(user_id):
    # 签名算法
    header = {'alg': 'HS256'}
    # 用于签名的密钥
    key = Cfg.get("secret")
    # 待签名的数据负载
    data = {'id': user_id}
    token = jwt.encode(header=header, payload=data, key=key).decode()

    # 重复登录
    v, meta = etcd.get(key=Cfg.get("token_dir") + token)
    if v is not None:
        lease_id = int.from_bytes(bytes=v, byteorder='big')
        etcd.refresh_lease(lease_id=lease_id)
    else:
        # token有效期
        new_lease = etcd.lease(ttl=Cfg.get("token_ttl"))
        etcd.put(Cfg.get("token_dir") + token, new_lease.id.to_bytes(8, byteorder='big'), lease=new_lease)
    return token


def get_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        ip = st.getsockname()[0]
    except socket.error:
        ip = '127.0.0.1'
    finally:
        st.close()
    return ip


def is_training():
    v, meta = etcd.get(key=Cfg.get('train_dir') + get_ip())
    if v is None or v.encode() != get_ip():
        return False
    return True


def offload_ip():
    etcd.delete(key=Cfg.get("server_dir") + Cfg.get("model") + "/" + get_ip())


def register_ip():
    global etcd
    lease_id = int.from_bytes(socket.inet_pton(socket.AF_INET, get_ip()), byteorder='big')
    try:
        lease = etcd.lease(ttl=5, lease_id=lease_id)
    except etcd3.exceptions.Etcd3Exception:
        print("申请lease失败。leaseID=", lease_id)
        sys.exit(0)
    else:
        etcd.put(key=Cfg.get("server_dir") + Cfg.get("model") + "/" + get_ip(),
                 value=Cfg.get("model") + "/" + get_ip(),
                 lease=lease,
                 prev_kv=False)
    # 隔1秒刷新
    while True:
        # 判断是否在训练：主动移除了IP
        if is_training():
            time.sleep(5)
        else:
            try:
                lease.refresh()
            except etcd3.exceptions.Etcd3Exception:
                # 重连etcd
                new_etcd = connect_etcd(allow_fail=True)
                if new_etcd is not None:
                    etcd = new_etcd

                    try:
                        lease = etcd.lease(ttl=5, lease_id=lease_id)
                    except etcd3.exceptions.Etcd3Exception:
                        print("重连etcd后，申请lease失败。leaseID=", lease_id)
                    else:
                        etcd.put(key=Cfg.get("server_dir") + Cfg.get("model") + "/" + get_ip(),
                                 value=Cfg.get("model") + "/" + get_ip(),
                                 lease=lease,
                                 prev_kv=False)
            time.sleep(1)


threading.Thread(target=register_ip).start()

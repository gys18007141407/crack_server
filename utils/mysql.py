import enum
from .config import Cfg
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.mysql import MEDIUMBLOB
from datetime import datetime

# 声明一个连接
engine = create_engine(f'mysql+pymysql://{Cfg.get("mysql_user")}:{Cfg.get("mysql_password")}@{Cfg.get("mysql_ip")}:{Cfg.get("mysql_port")}/{Cfg.get("mysql_database")}?charset=utf8', echo=True, future=True)
# 构建session，跟数据库操作只能用session
session_factory = sessionmaker(bind=engine)  # 构建Session会话类

# 声明ORM的一个基类并建立映射关系
Base = declarative_base()


class PermissionEnum(enum.Enum):
    Admin = 1
    User = 2


class ImageTypeEnum(enum.Enum):
    Undefined = 1
    Train = 2
    Validation = 3
    Test = 4


class Permission(Base):
    """权限表"""
    __tablename__ = 'permission'
    id = Column(Integer, name='id', primary_key=True)
    role = Column(Enum(PermissionEnum), nullable=False, name='role', comment='role', unique=True)
    desc = Column(String(256), nullable=False, name='desc', comment="desc", unique=True)

    def __repr__(self):
        return '<{} id={} role={} desc={}>'.format(self.__class__.__name__, self.id, self.role, self.desc)


class User(Base):
    """用户表"""
    __tablename__ = 'user'
    id = Column(Integer, name='id', primary_key=True)
    name = Column(String(32), nullable=False, name="name", comment="name", unique=True)
    password = Column(String(512), nullable=False, name="password", comment="password")
    permission = Column(Integer, ForeignKey('permission.id'), name="permission")
    create_at = Column(DateTime, default=datetime.now(), name='create_at', comment="create_at")
    update_at = Column(DateTime, onupdate=datetime.now(), default=datetime.now(),  name='update_at', comment="update_at")
    relation = relationship('Permission', backref='permission2user', uselist=False)  # 外键：角色

    def __repr__(self):
        return '<{} id={} name={} password={}>'.format(self.__class__.__name__, self.id, self.name, self.password)


class CrackDatasets(Base):
    """数据集表"""
    __tablename__ = 'crack_datasets'
    id = Column(Integer, name='id', primary_key=True)
    name = Column(String(256), nullable=False, name='name', comment="name", unique=True)
    desc = Column(String(256), nullable=False, name='desc', comment="desc")

    def __repr__(self):
        return '<{} id={} name={}>'.format(self.__class__.__name__, self.id, self.name)


class CrackImagesType(Base):
    """训练集、验证集、测试集"""
    __tablename__ = 'crack_images_type'
    id = Column(Integer, name='id', primary_key=True)
    type = Column(Enum(ImageTypeEnum), nullable=False, unique=True, name='type')
    desc = Column(String(256), nullable=False, name='desc', comment="desc", unique=True)

    def __repr__(self):
        return '<{} id={} type={} desc={}>'.format(self.__class__.__name__, self.id, self.type, self.desc)


class CrackImages(Base):
    """图片表"""
    __tablename__ = 'crack_images'
    id = Column(Integer, name='id', primary_key=True)
    name = Column(String(32), nullable=False, name='name', comment="name", unique=True)
    image = Column(MEDIUMBLOB, nullable=False, name='image', comment='image')
    label = Column(MEDIUMBLOB, nullable=False, name='label', comment='label')
    datasets = Column(Integer, ForeignKey('crack_datasets.id'))
    type = Column(Integer, ForeignKey('crack_images_type.id'))
    create_at = Column(DateTime, default=datetime.now(), name='create_at', comment="create_at")
    update_at = Column(DateTime, onupdate=datetime.now(), default=datetime.now(), name='update_at', comment="update_at")
    relation1 = relationship('CrackDatasets', backref='datasets2imgs', uselist=False)
    relation2 = relationship('CrackImagesType', backref='imgtypes2imgs', uselist=False)

    def __repr__(self):
        return '<{} id={} name={}>'.format(self.__class__.__name__, self.id, self.name)


def init_mysql_tables():
    global Base
    global engine
    # 同步表
    Base.metadata.create_all(engine)  # 创建表


def test_mysql():
    # 以下为测试代码
    session = session_factory()  # 实例化，得到session实例对象
    # 增加角色
    admin = Permission(role=PermissionEnum.Admin, desc='admin')
    user = Permission(role=PermissionEnum.User, desc='user')
    session.add(admin)
    session.add(user)
    session.commit()
    print(session.query(Permission).all())

    # 添加用户
    import hashlib
    gys_id = session.query(Permission.id).filter(Permission.role == PermissionEnum.Admin).first().id
    b1102_id = session.query(Permission.id).filter(Permission.role == PermissionEnum.User).first().id
    obj = hashlib.sha256()
    obj.update('123456'.encode())
    obj.hexdigest()
    gys = User(name='gys', password=obj.hexdigest(), permission=gys_id)
    obj = hashlib.sha256()
    obj.update('b1102b1102'.encode())
    b1102 = User(name='b1102', password=obj.hexdigest(), permission=b1102_id)
    session.add(gys)
    session.commit()
    print(session.query(User).all())

    # 添加数据集
    CFD_s160_d160 = CrackDatasets(name="CFD_s160_d160", desc="generated from CFD")
    session.add(CFD_s160_d160)
    session.commit()
    print(session.query(CrackDatasets).all())

    # 添加图片类型
    Undefined = CrackImagesType(type=ImageTypeEnum.Undefined, desc='undefined')
    Train = CrackImagesType(type=ImageTypeEnum.Train, desc='train')
    Validation = CrackImagesType(type=ImageTypeEnum.Validation, desc='validation')
    Test = CrackImagesType(type=ImageTypeEnum.Test, desc='test')
    session.add(Undefined)
    session.add(Train)
    session.add(Validation)
    session.add(Test)
    session.commit()
    print(session.query(CrackImagesType).all())

    # 添加图片
    img_dir = r'D:\net-design\predict_result\tmp\test_imgs'
    label_dir = r'D:\net-design\predict_result\tmp\test_labels'
    import os
    imgs = os.listdir(img_dir)
    imgs.sort(key=lambda x: x[:x.rfind(".")])

    labels = os.listdir(label_dir)
    labels.sort(key=lambda x: x[:x.rfind(".")])

    datasets_id = session.query(CrackDatasets.id).filter(CrackDatasets.name == 'CFD_s160_d160').first().id
    type_id = session.query(CrackImagesType.id).filter(CrackImagesType.type == ImageTypeEnum.Test).first().id
    for img, label in zip(imgs, labels): #img name和 label name是一样的
        with open(os.path.join(img_dir, img), 'rb') as f:
            img_bytes = f.read()
        with open(os.path.join(label_dir, label), 'rb') as f:
            label_bytes = f.read()

        session.add(CrackImages(name=img, image=img_bytes, label=label_bytes, datasets=datasets_id, type=type_id))
    session.commit()
    print(session.query(CrackImages).count())


    # 读取图片
    img = session.query(CrackImages).first()
    with open(img.name, 'wb') as f:
        f.write(img.label)
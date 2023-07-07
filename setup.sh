#!/bin/bash

# 运行脚本前的准备操作：
# 保证setup.sh与crack_server和crack_web都在~目录下

# 如何运行该脚本？
# 格式：sh setup.sh [root的password]
# 例如：sh setup.sh 123456

# root用户的密码
PASSWORD=$1  # 作为参数传入，或者直接在这里修改，如PASSWORD="123456"
if [ ! -n "${PASSWORD}" ]
then
        echo "lack param: epoch="
        exit 0
fi


# 进入home目录
cd ~

#ARCH=x86_64
#ARCH=aarch64
ARCH=`arch`
anaconda_url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2022.10-Linux-${ARCH}.sh"

echo "arch=${ARCH}"

# 安装conda
if [ ! `which conda` ]
then
  wget --user-agent="Mozilla" ${anaconda_url}
  bash Anaconda3-2022.10-Linux-${ARCH}.sh
  . anaconda3/bin/activate
  conda init
fi

# 创建虚拟环境
if [ `conda env list | grep crack_env | awk '{print $1}'` ]
then
  echo 'conda env existed!'
else
  echo "conda env create start!"
  conda create -n crack_env python=3.7
  echo "conda env create success!"
fi

# 激活虚拟环境
eval "$(conda shell.bash hook)"
conda activate crack_env
echo "current conda env is: $CONDA_DEFAULT_ENV"

# 安装python依赖
echo "python requiremets install start!"
pip install -r ./crack_server/requirements.txt
echo "python requiremets install success!"

# 安装nodejs 18.16.1
if [ ! `which npm` ]
then
  echo "nodejs install start!"
  wget https://npmmirror.com/mirrors/node/v18.16.1/node-v18.16.1-linux-x64.tar.xz
  echo ${PASSWORD} | sudo -S tar xf node-v18.16.1-linux-x64.tar.xz -C /usr/local/
  echo ${PASSWORD} | sudo -S mv /usr/local/node-v18.16.1-linux-x64/ /usr/local/nodejs
  echo ${PASSWORD} | sudo -S ln -s /usr/local/nodejs/bin/node /usr/local/bin
  echo ${PASSWORD} | sudo -S ln -s /usr/local/nodejs/bin/npm /usr/local/bin
  echo "nodejs install success!"
fi

# 安装js依赖
cd crack_web
npm install
cd ..

# 脚本运行结束后如何启动系统？
# 前端，进入crack_web目录运行：npm run serve -- --port 13102
# 后端，进入crack_server目录运行：sudo ~/anaconda3/envs/crack_env/bin/python3.7 run.py


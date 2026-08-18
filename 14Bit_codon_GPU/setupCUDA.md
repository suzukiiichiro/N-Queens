# install pyenv 
curl -fsSL https://pyenv.run | bash

# ~/.bash_profile pyenv settings
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# pyenvでpythonのインストール
$ CFLAGS=-I/usr/include/openssl LDFLAGS=-L/usr/lib pyenv install -v 3.13.5

#  うまくいかないときは作りかけのバイナリを削除
pyenv uninstall -f 3.13.5

#  pyenv
NQueens2$ pyenv global 3.13.5
NQueens2$ pyenv versions
  system
  * 3.13.5 (set by /home/ec2-user/.pyenv/version)
  NQueens2$

# pip install
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

#  cuda
$ nvidia-smi            # ドライバとCUDA Runtimeのバージョンを確認
$ nvcc --version        # (入っていれば) CUDA Toolkitのバージョン確認

# ① まず（もし入っていたら）ソース版 cupy を外す
python -m pip uninstall -y cupy

# ② pip を新しめに
python -m pip install -U pip setuptools wheel

# ③ CUDA 12.x 用の wheel を "必ず" 取る（ビルドさせない）
python -m pip install --only-binary=:all: -U cupy-cuda12x


#########################
# CUDA ルートを自動検出　このままターミナルに貼る
CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
# ランタイム&コンパイラ
echo "export CUDA_PATH=${CUDA_ROOT}"                   >> ~/.bashrc
echo 'export PATH="$CUDA_PATH/bin:$PATH"'             >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc
# ★ NVRTC 用の include パス（これがポイント）
echo 'export CPATH="$CUDA_PATH/targets/aarch64-linux/include:${CPATH}"' >> ~/.bashrc
echo 'export CUPY_NVRTC_INCLUDE_DIRS="$CUDA_PATH/targets/aarch64-linux/include"' >> ~/.bashrc
# （もし sbsa-linux だったら aarch64-linux を sbsa-linux に置換）
# 反映
exec $SHELL -l
#########################


# 確認
python -m pip show cupy-cuda12x || \
python -m pip install --only-binary=:all: -U cupy-cuda12x
# 無印 cupy が入ってたらアンインストール
python -m pip uninstall -y cupy


# 確認
python - <<'PY'
import cupy as cp, cupy
print("CUDA_PATH:", cupy.cuda.get_cuda_path())
print("Runtime ver:", cupy.cuda.runtime.runtimeGetVersion())
x = cp.arange(8, dtype=cp.float16)  # fp16 を含む ufunc で NVRTC にヘッダが要る
print("sum:", (x + 1).sum())
cupy.show_config()
PY




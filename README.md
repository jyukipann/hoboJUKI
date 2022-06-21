# hoboJUKI
ほぼじゅきを作るぞー! 2022/6/20に開発を開始しました。

まずは自分のデータを集める。

そしてデータセットとして成形

学習

モデル完成

チャットボットとしての体裁を整える

## 環境構築
conda 環境を構築する。pythonバージョンは3.7として、極力conda installで行うが無理な場合はpip installを使う。開発で使うメインPCのcudaバージョンは11.1だった。
```
conda create -n hoboJUKI python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge transformers
conda install -c anaconda tensorflow-gpu
conda install -c conda-forge tensorboard
conda install -c conda-forge sentencepiece
conda install -c conda-forge tqdm
pip install fugashi
pip install fugashi[unidic]
python -m unidic download
```
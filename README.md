# hoboJUKI
ほぼじゅきを作るぞー! 2022/6/20に開発を開始しました。

まずは自分のデータを集める。

そしてデータセットとして成形

学習

モデル完成

チャットボットとしての体裁を整える

## 環境構築
conda 環境を構築する。pythonバージョンは3.10として、すべてpipでインストールした
まずhttps://huggingface.co/rinna/japanese-gpt-1b/tree/mainをクローンした。これもクローンhttps://github.com/rinnakk/japanese-pretrained-models。
```
git lfs install
conda create -n hoboJUKI
conda activate hoboJUKI
cd pathto/japanese-pretrained-models
pip install -r requirements.txt
python -m unidic download
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers
pip install sentencepiece
```

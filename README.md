# hoboJUKI
ほぼじゅきを作るぞー! 2022/6/20に開発を開始しました。

まずは自分のデータを集める。

そしてデータセットとして成形

学習

モデル完成

チャットボットとしての体裁を整える

## 環境構築
conda 環境を構築する。pythonバージョンは3.10として、すべてpipでインストールした
まず https://huggingface.co/rinna/japanese-gpt-1b/tree/main をクローンした。これもクローン https://github.com/rinnakk/japanese-pretrained-models 
```
git lfs install
conda create -n hoboJUKI  pytho=3.10
conda activate hoboJUKI
pip install -r requirements.txt
python -m unidic download
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers
pip install sentencepiece
```
## メモ
https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling

https://github.com/npaka3/akane-talk/blob/main/docs/dataset.txt

https://note.com/npaka/n/n8a435f0c8f69

チャットデータの収集はほぼ完了なので、テキストを整形していく。どのような形式がいいのかを調査中。
# hoboJUKI
ほぼじゅきを作るぞー! 2022/6/20に開発を開始しました。

まずは自分のデータを集める。

そしてデータセットとして成形

学習

モデル完成

チャットボットとしての体裁を整える

# 環境構築
conda 環境を構築する。pythonバージョンは3.10として、すべてpipでインストールした
まず https://huggingface.co/rinna/japanese-gpt-1b/tree/main をクローンした。これもクローン https://github.com/rinnakk/japanese-pretrained-models 
```cmd
git lfs install
conda create -n hoboJUKI  python=3.10
conda activate hoboJUKI
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install git+https://github.com/huggingface/peft.git
pip install accelerate bitsandbytes
pip install scipy
pip install sentencepiece

pip install -r requirements.txt
python -m unidic download
pip install git+https://github.com/Rapptz/discord.py.git
```
これもやった。trainでこれがないと怒られる。まじでこれやれば動きました。

```
J: & cd J:\workspace\GitHub\hoboJUKI & conda activate hoboJUKI
```

# メモ
https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling
./transformers/examples/language-modeling/run_clm.py」

https://github.com/npaka3/akane-talk/blob/main/docs/dataset.txt

https://note.com/npaka/n/n8a435f0c8f69

チャットデータの収集はほぼ完了なので、テキストを整形していく。どのような形式がいいのかを調査中。
    とりあえず、ツイートを一行ずつ書いたテキストにした。

https://rooter.jp/ml/japanese-ad-text-generate/


```
python train/test_train.py \
    --model_name_or_path=rinna/japanese-gpt-1b \
    --train_file=dataset/tweet.txt \
    --validation_file=dataset/tweet.txt \
    --do_train \
    --do_eval \
    --num_train_epochs=1 \
    --save_steps=5000 \
    --save_total_limit=3 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir=output/ \
    --use_fast_tokenizer=False
```
```
python train/test_train.py --model_name_or_path=rinna/japanese-gpt-1b --train_file=dataset/tweet.txt --validation_file=dataset/tweet.txt --do_train --do_eval --num_train_epochs=1 --save_steps=5000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=output/ --use_fast_tokenizer=False --overwrite_output_dir
```

```
python train/test_train.py --model_name_or_path=rinna/japanese-gpt-1b --train_file=dataset/train.txt --validation_file=dataset/train.txt --do_train --do_eval --num_train_epochs=10 --save_steps=5000 --save_total_limit=3 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --output_dir=output_v1/ --use_fast_tokenizer=False --overwrite_output_dir
```

ちなみに、japanese GPT-1b は、RTX 2070 8GB じゃ動きませんでした。
RTX 3090 24GB でも動きませんでした。
RTX A6000 48GB で試しています。動きました。

# 実行方法
```cmd
cd W:\tanimoto.j\workspace\GitHub\hoboJUKI
W:
conda activate hoboJUKI
python discordbot\hoboJUKI_discord.py
```
```
cd W:\tanimoto.j\workspace\GitHub\hoboJUKI & W: & conda activate hoboJUKI & python discordbot\hoboJUKI_discord.py
```

# アイデア
## 前回は学習回しすぎ
epoch300は多すぎ。val用のデータセットは必要。

## 明示的に無視したい
無視することで会話を終わらせる。次に会話が続くかを推論する必要がある。
データセットの変更は必須。

## 再学習の方法
データセットどうする？


## 進捗（2023/07/15）
[qiitaに寄稿した記事](https://qiita.com/jyukipann/items/07c7b156b2dc5a126b30)でhoboJUKIのモデルを学習しました。
まぁ考えなきゃいけないことはまだまだあるけど、程よくいい感じにモデルは出来た。
48時間程度A6000を回した。
`hoboJUKI-3.6b/source`にbotのコードを再構築していく。
discordpyを最新にしたいので、破壊的な環境操作が起きるかもしれない。
```cmd.exe
pip install update discord.py
```

(これ)[https://qiita.com/selllous/items/fbaa2c3d2d504e436b17]でbitsandbytesやるしかないかな。
https://qiita.com/selllous/items/c4880767da45173227c1

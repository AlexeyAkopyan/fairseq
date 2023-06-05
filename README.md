# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* To install:

``` bash
git clone https://github.com/AlexeyAkopyan/transformer_modification
cd transformer_modification
pip install --editable ./
```

# Training models on En-Fr 

Download and prepare the data
``` bash
cd examples/translation/
bash prepare-europarl-v7-en2fr.sh
cd ../..
```

Binarize the dataset
``` bash
TEXT=examples/translation/En_Fr
DIR=Europart-v7-En2Fr
mkdir -p DIR

fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --testpref $TEXT/test --destdir $DIR/data/binarized \
    --joined-dictionary --workers 10
```

Train the models

``` bash
# Classic Transformer
fairseq-train
    $DIR\data\binarized 
    --arch transformer_cascade_wmt_en_fr_big 
    --encoder-self-att-heads 16 --decoder-self-att-heads 16 --decoder-cross-att-heads 16 -s en -t fr 
    --share-all-embeddings --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 0.0005 
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.1 --weight-decay 0.0 
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 500 --update-freq 7 
    --eval-bleu --best-checkpoint-metric bleu  --maximize-best-checkpoint-metric --seed 42 
    --validate-interval-updates 2000 --save-dir $DIR\checkpoints\classic --fp16
   
# Modified Transformer
fairseq-train 
    $DIR\data\binarized
    --arch transformer_cascade_wmt_en_fr_big 
    --encoder-self-att-heads 16 --decoder-self-att-heads 16 --decoder-cross-att-heads 0,0,8,16,24,48 -s en -t fr 
    --share-all-embeddings --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 0.0005 
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.1 --weight-decay 0.0 
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 500 --update-freq 7 
    --eval-bleu --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --seed 42 
    --validate-interval-updates 2000 --save-dir $DIR\checkpoints\modified --fp16
```

Generate and detokenize test translations
``` bash 
# Classic Transformer
fairseq-generate
    $DIR/data/binarized 
    --path $DIR\checkpoints\classic\checkpoint_best.pt 
    --batch-size 64 --remove-bpe --gen-subset test --source-lang en --target-lang fr 
    --scoring sacrebleu --sacrebleu --beam 4 --lenpen 0.6 
    --results-path $DIR/generated/classic
    
# Modified Transformer
fairseq-generate
    $DIR/data/binarized 
    --path $DIR\checkpoints\modified\checkpoint_best.pt 
    --batch-size 64 --remove-bpe --gen-subset test --source-lang en --target-lang fr 
    --scoring sacrebleu --sacrebleu --beam 4 --lenpen 0.6 
    --results-path $DIR/generated/modified
    
bash scripts/detokenize_en2fr.sh 
```
Score generated translations
``` bash
pip install sacrebleu
GEN=$DIR/generated

sacrebleu $GEN/classic/true-detok.txt -i $GEN/classic/gen-detok.txt -m bleu -b -w 4 
sacrebleu $GEN/classic/true-detok.txt -i $GEN/modified/gen-detok.txt -m bleu -b -w 4 
```




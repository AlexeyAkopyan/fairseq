# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* To install:

``` bash
git clone https://github.com/AlexeyAkopyan/transformer_modification
cd transformer_modification
pip install --editable ./
pip install sacrebleu
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
mkdir -p $DIR

fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --testpref $TEXT/test --destdir $DIR/data/binarized \
    --joined-dictionary --workers 10
```

Train the models

``` bash
# Classic Transformer
fairseq-train \
    $DIR\data\binarized \ 
    --arch transformer_cascade_wmt_en_fr_big \ 
    --encoder-self-att-heads 16 --decoder-self-att-heads 16 --decoder-cross-att-heads 16 -s en -t fr \ 
    --share-all-embeddings --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 0.0005 \ 
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.1 --weight-decay 0.0 \ 
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 500 --update-freq 7 \ 
    --eval-bleu --best-checkpoint-metric bleu  --maximize-best-checkpoint-metric --seed 42 \ 
    --validate-interval-updates 2000 --save-dir $DIR\checkpoints\classic --fp16 
   
# Modified Transformer
fairseq-train \
    $DIR\data\binarized \
    --arch transformer_cascade_wmt_en_fr_big \ 
    --encoder-self-att-heads 16 --decoder-self-att-heads 16 --decoder-cross-att-heads 0,0,8,16,24,48 -s en -t fr \ 
    --share-all-embeddings --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 0.0005 \ 
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.1 --weight-decay 0.0 \ 
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 500 --update-freq 7 \ 
    --eval-bleu --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --seed 42 \ 
    --validate-interval-updates 2000 --save-dir $DIR\checkpoints\modified --fp16
```

Generate and detokenize test translations
``` bash 
GEN=$DIR/generated

# Classic Transformer
fairseq-generate\
    $DIR/data/binarized \
    --path $DIR\checkpoints\classic\checkpoint_best.pt \
    --batch-size 64 --remove-bpe --gen-subset test --source-lang en --target-lang fr \
    --scoring sacrebleu --sacrebleu --beam 4 --lenpen 0.6 \
    --results-path $GEN/classic
    
# Modified Transformer
fairseq-generate \
    $DIR/data/binarized \
    --path $DIR\checkpoints\modified\checkpoint_best.pt \
    --batch-size 64 --remove-bpe --gen-subset test --source-lang en --target-lang fr \
    --scoring sacrebleu --sacrebleu --beam 4 --lenpen 0.6 \
    --results-path $GEN/modified
    
bash scripts/detokenize.sh $DIR en fr
```

Score generated translations
``` bash
sacrebleu $GEN/classic/true-detok.txt -i $GEN/classic/gen-detok.txt -m bleu -b -w 4 
sacrebleu $GEN/classic/true-detok.txt -i $GEN/modified/gen-detok.txt -m bleu -b -w 4 
```

# Training models on Ru-En

Download and prepare the data
``` bash
cd examples/translation/
bash prepare-wmt16ru2en.sh
cd ../..
```

Binarize the dataset
``` bash
TEXT=examples/translation/Ru_En
DIR=Ru2En
mkdir -p $DIR

fairseq-preprocess \
    -s en -t fr --trainpref $TEXT/train --validpref $TEXT/valid --destdir $DIR/data/binarized \
    --joined-dictionary --workers 10
    
TEST=$TEXT/newstest
fairseq-preprocess -s ru -t en --testpref $TEST2014 --destdir $DIR/data/binarized/newstest2014 --workers 10
fairseq-preprocess -s ru -t en --testpref $TEST2015 --destdir $DIR/data/binarized/newstest2015 --workers 10
fairseq-preprocess -s ru -t en --testpref $TEST2016 --destdir $DIR/data/binarized/newstest2016 --workers 10
fairseq-preprocess -s ru -t en --testpref $TEST2017 --destdir $DIR/data/binarized/newstest2017 --workers 10
fairseq-preprocess -s ru -t en --testpref $TEST2018 --destdir $DIR/data/binarized/newstest2018 --workers 10
```

Train the models

``` bash
# Classic Transformer
fairseq-train \
    $DIR\data\binarized 
    --arch transformer_cascade_wmt_en_de_big \
    --encoder-self-att-heads 16 --decoder-self-att-heads 16 --decoder-cross-att-heads 16 -s ru -t en \
    --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 0.0007 --warmup-updates 4000 \
    --lr-scheduler inverse_sqrt --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 1024 --update-freq 8 --seed 42 --log-interval 100 \
    --save-dir $DIR\checkpoints\classic --fp16
   
# Modified Transformer
fairseq-train \
    $DIR\data\binarized \
    --arch transformer_cascade_wmt_en_de_big \
    --encoder-self-att-heads 16 --decoder-self-att-heads 16 --decoder-cross-att-heads 0,0,8,16,24,48 -s ru -t en \
    --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 0.0007 --warmup-updates 4000 \ 
    --lr-scheduler inverse_sqrt --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \ 
    --label-smoothing 0.1 --max-tokens 1024 --update-freq 8 --seed 42 --log-interval 100 \ 
    --save-dir $DIR\checkpoints\modified --fp16
```
Generate and detokenize test translations
``` bash 
GEN=$DIR/generated

# Classic Transformer
fairseq-generate\
    $DIR/data/binarized/newstest2014 \
    --path $DIR\checkpoints\classic\checkpoint_best.pt \
    --batch-size 64 --remove-bpe --gen-subset test --source-lang ru --target-lang en \
    --scoring sacrebleu --sacrebleu --beam 4 --lenpen 0.6 \
    --results-path $GEN/classic
    
# Modified Transformer
fairseq-generate \
    $DIR/data/binarized/newstest2014 \
    --path $DIR\checkpoints\modified\checkpoint_best.pt \
    --batch-size 64 --remove-bpe --gen-subset test --source-lang ru --target-lang en \
    --scoring sacrebleu --sacrebleu --beam 4 --lenpen 0.6 \
    --results-path $GEN/modified
    
bash scripts/detokenize.sh $DIR ru en
```
Score generated translations
``` bash
sacrebleu $GEN/classic/true-detok.txt -i $GEN/classic/gen-detok.txt -m bleu -b -w 4 
sacrebleu $GEN/classic/true-detok.txt -i $GEN/modified/gen-detok.txt -m bleu -b -w 4 
```




#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_EN_TOKENS=28000
BPE_RU_TOKENS=28000

URLS=(
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt20/translation-task/dev.tgz"
)
FILES=(
    "training-parallel-commoncrawl.tgz"
    "dev.tgz"
)
CORPORA=(
    "corpus.en_ru.1m"
    "commoncrawl.ru-en"
)
NEWSTEST_YEARS=(
    4
    5
    6
    7
    8
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=ru
tgt=en
lang=ru-en
prep=Ru_En
tmp=$prep/tmp
orig=orig

mkdir -p $prep $orig $tmp

cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for y in "${NEWSTEST_YEARS[@]}"; do
    for l in $src $tgt; do
        if [ "$l" == "$src" ]; then
            t="src"
        else
            t="ref"
        fi
        grep '<seg id' $orig/dev/newstest201$y-ruen-$t.$l.sgm | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -a -l $l > $tmp/newstest201$y.$l
        echo ""
    done
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

BPE_EN_CODE=$prep/en_code
BPE_RU_CODE=$prep/ru_code

echo "learn_bpe.py on train.en..."
python $BPEROOT/learn_bpe.py -s $BPE_EN_TOKENS < $tmp/train.en > $BPE_EN_CODE
echo "learn_bpe.py on train.ru..."
python $BPEROOT/learn_bpe.py -s $BPE_RU_TOKENS < $tmp/train.ru > $BPE_RU_CODE

for f in train.en valid.en; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_EN_CODE < $tmp/$f > $tmp/bpe.$f
done
for f in train.ru valid.ru; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_RU_CODE < $tmp/$f > $tmp/bpe.$f
done

for y in "${NEWSTEST_YEARS[@]}"; do
    f=$tmp/newstest201$y.en
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_EN_CODE < $tmp/$f > $tmp/bpe.$f
done

for y in "${NEWSTEST_YEARS[@]}"; do
    f=$tmp/newstest201$y.ru
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_RU_CODE < $tmp/$f > $tmp/bpe.$f
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.valid.$L $prep/valid.$L
done

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done

for y in "${NEWSTEST_YEARS[@]}"; do
    for L in $src $tgt; do
        cp $tmp/bpe.newstest201$y.$L $prep/newstest201$y.$L
    done
done
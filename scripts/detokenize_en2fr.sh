echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
DETOKENIZER=mosesdecoder/scripts/tokenizer/detokenizer.perl

DIR=Europart-v7-En2Fr

echo 'Split on sys and ref'
grep ^S $DIR/generated/classic/generate-test.txt | cut -f2- > source-bpe.txt
grep ^T $DIR/generated/classic/generate-test.txt | cut -f2- > true-bpe.txt
grep ^H $DIR/generated/classic/generate-test.txt | cut -f3- > classic-bpe.txt
grep ^H $DIR/generated/modified/generate-test.txt | cut -f3- > modified-bpe.txt

echo 'Start decode bpe'
cat source-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > source-tok.txt
cat true-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > true-tok.txt
cat classic-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > classic-tok.txt
cat modified-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > modified-tok.txt

echo 'Start detokenize'
perl $DETOKENIZER -l en < source-tok.txt > $DIR/generated/classic/source-detok.txt
perl $DETOKENIZER -l fr < true-tok.txt > $DIR/generated/classic/true-detok.txt
perl $DETOKENIZER -l fr < classic-tok.txt > $DIR/generated/classic/gen-detok.txt
perl $DETOKENIZER -l fr < modified-tok.txt > $DIR/generated/modified/gen-detok.txt

echo 'Remove auxiliary files'
rm classic-bpe.txt true-bpe.txt modified-bpe.txt classic-tok.txt true-tok.txt modified-tok.txt
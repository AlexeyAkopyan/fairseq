echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
DETOKENIZER=mosesdecoder/scripts/tokenizer/detokenizer.perl

echo 'Split on sys and ref'
grep ^S $1/generated/classic/generate-test.txt | cut -f2- > source-bpe.txt
grep ^T $1/generated/classic/generate-test.txt | cut -f2- > true-bpe.txt
grep ^H $1/generated/classic/generate-test.txt | cut -f3- > classic-bpe.txt
grep ^H $1/generated/modified/generate-test.txt | cut -f3- > modified-bpe.txt

echo 'Start decode bpe'
cat source-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > source-tok.txt
cat true-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > true-tok.txt
cat classic-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > classic-tok.txt
cat modified-bpe.txt | sed -r 's/(@@ )|(@@ ?$)|( @)|(@ )//g' > modified-tok.txt

echo 'Start detokenize'
perl $DETOKENIZER -l $2 < source-tok.txt > $1/generated/classic/source-detok.txt
perl $DETOKENIZER -l $3 < true-tok.txt > $1/generated/classic/true-detok.txt
perl $DETOKENIZER -l $3 < classic-tok.txt > $1/generated/classic/gen-detok.txt
perl $DETOKENIZER -l $3 < modified-tok.txt > $1/generated/modified/gen-detok.txt

echo 'Remove auxiliary files'
rm classic-bpe.txt true-bpe.txt modified-bpe.txt classic-tok.txt true-tok.txt modified-tok.txt
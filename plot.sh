
DOT="treepic.dot"
PNG="treepic"
MODEL="model"

a="_1"
b="_2"
c="_3"

python main.py --task plot --load $MODEL --plot $DOT
dot -T png $DOT$a -o "${PNG}${a}.png"
dot -T png $DOT$b -o "${PNG}${b}.png"
dot -T png $DOT$c -o "${PNG}${c}.png"

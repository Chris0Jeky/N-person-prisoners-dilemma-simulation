#set output 'avgQ.gif'
set output 'avgQ.tex'
#set terminal latex
#set terminal gif
set terminal epslatex color

set palette color


set xlabel "Round"
set ylabel "Cooperation"

#set origin -0.07, -0.1
set offsets 0.0, 0.0, 0.25,0.0

set title "Average Cooperation by Round with Q Learning Policies"

plot 'qLearningTFT-E.txt' using 1:5 title ' pair 2Q+1TFT-E-Q' with lines, \
     'qLearningTFT-E.txt' using 1:4 title ' pair 2Q+1TFT-E-T' with lines, \
     'qLearningTFT-E.txt' using 1:8 title ' neighbourhood 2Q+1TFT-E-Q' with lines, \
     'qLearningTFT-E.txt' using 1:9 title ' neighbourhood 2Q+1TFT-E-T' with lines


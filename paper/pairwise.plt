#set output 'pairwise.gif'
set output 'pairwise.tex'
#set terminal latex
#set terminal gif
set terminal epslatex color 8

set palette color

set size 0.7,0.7
#set key font "Arial,8"


set xlabel "Round"
set ylabel "Cooperation"

#set origin -0.07, -0.1
set offsets 0.0, 0.0, 0.25,0.0

set title "Average Cooperation by Round with Pairwise Voting"

plot 'pairwiseData.txt' using 1:2 title '3-TFT' with lines, \
     'pairwiseData.txt' using 1:4 title '2-TFT+D' with lines, \
     'pairwiseData.txt' using 1:3 title '2-TFT-E+D' with lines, \
     'pairwiseData.txt' using 1:5 title '2-TFT-E+C' with lines 

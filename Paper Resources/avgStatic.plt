#set output 'avg.gif'
set output 'avg.tex'
#set terminal latex
#set terminal gif
set terminal epslatex color

set palette color


set xlabel "Round"
set ylabel "Cooperation"

#set origin -0.07, -0.1
set offsets 0.0, 0.0, 0.25,0.0

set title "Average Cooperation by Round with Static Policies"

plot 'avgStatic100.txt' using 1:2 title 'neighbourhood 2TFT-E+C' with lines, \
     'avgStatic100.txt' using 1:4 title 'pair 2TFT-E+C' with lines, \
     'avgStatic100.txt' using 1:5 title 'pair 2TFT-E+D' with lines, \
     'avgStatic100.txt' using 1:3 title 'neighbourhood 2TFT-E+D' with lines


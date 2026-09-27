#set output 'neighbourhood.gif'
set output 'neighbourhood.tex'
#set terminal latex
#set terminal gif
set terminal epslatex color

set palette color


set xlabel "Round"
set ylabel "Cooperation"

#set origin -0.07, -0.1
set offsets 0.0, 0.0, 0.25,0.0

set title "Average Cooperation by Round with Neighbourhood Voting"

plot 'neighbourhoodData.txt' using 1:2 title '3-TFT' with lines, \
     'neighbourhoodData.txt' using 1:4 title '2-TFT+D' with lines, \
     'neighbourhoodData.txt' using 1:3 title '2-TFT-E+D' with lines, \
     'neighbourhoodData.txt' using 1:5 title '2-TFT-E+C' with lines 

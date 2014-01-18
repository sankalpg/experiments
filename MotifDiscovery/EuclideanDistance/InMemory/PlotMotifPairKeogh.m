function [] = PlotMotifPairKeogh(pitchfile, outputfile, samples)

data = dlmread(pitchfile);

motifpair = dlmread(outputfile);

for i=1:size(motifpair,1)

motif1= data(motifpair(i,1)+1:motifpair(i,1)+samples);
motif2= data(motifpair(i,2)+1:motifpair(i,2)+samples);
figure; 
plot(motif1);
hold on
plot(motif2,'r');

distance = sqrt(sum(abs(motif1-motif2).^2))

end


end

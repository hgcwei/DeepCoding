function [ mat ] = hb_encoding( seq )
m1 = c2_encoding(seq);
m2 = codon64(seq);
mat = [m1,zeros(90,1)];
mat(1:64,3) = m2;
end

